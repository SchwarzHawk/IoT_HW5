import os
import re
import math
import numpy as np
import random
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction import text as sk_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# optional sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SB = True
except Exception:
    _HAS_SB = False


class TransformerEmbedder(BaseEstimator, TransformerMixin):
    """Wrap a sentence-transformers model to produce fixed-size embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        if self.model is None and _HAS_SB:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None
        return self

    def transform(self, X):
        if not _HAS_SB:
            # fall back to simple length-based features
            return np.array([[len(str(x))] for x in X])
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        emb = self.model.encode(list(X), convert_to_numpy=True)
        return emb


class EmbeddingStats(BaseEstimator, TransformerMixin):
    """Compute simple statistics over sentence embeddings.

    Returns a single numeric feature per document representing variance
    across sentence embeddings (if sentence-transformers is available),
    otherwise returns zero.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        if self.model is None and _HAS_SB:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None
        return self

    def transform(self, X):
        rows = []
        if not _HAS_SB or self.model is None:
            for _ in X:
                rows.append([0.0])
            return np.array(rows)

        for doc in X:
            try:
                sents = re.split(r"[.!?]+\s*", str(doc))
                sents = [s for s in sents if s.strip()]
                if len(sents) <= 1:
                    rows.append([0.0])
                    continue
                emb = self.model.encode(sents, convert_to_numpy=True)
                # variance across sentences (mean of variances per-dimension)
                var = float(np.mean(np.var(emb, axis=0)))
                rows.append([var])
            except Exception:
                rows.append([0.0])

        return np.array(rows)


# try to import transformers for optional perplexity feature
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# try to import HF pipeline for text-classification (optional fine-tuned models)
if _HAS_TRANSFORMERS:
    try:
        from transformers import pipeline as hf_pipeline
        _HAS_HF_PIPELINE = True
    except Exception:
        _HAS_HF_PIPELINE = False
else:
    _HAS_HF_PIPELINE = False

# try to import jieba for Chinese tokenization
try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


class TextStats(BaseEstimator, TransformerMixin):
    """Extract simple stylometric features from raw text.

    If `use_perplexity` is True and `transformers` is available, will compute a
    token-level perplexity using GPT-2 (can be slow). When perplexity is not
    enabled we return 0.0 for that feature (avoid NaNs).
    """

    def __init__(self, use_perplexity: bool = False):
        self.use_perplexity = use_perplexity and _HAS_TRANSFORMERS
        self.stop_words = sk_text.ENGLISH_STOP_WORDS
        self.gpt2_tokenizer = None
        self.gpt2_model = None
        if self.use_perplexity:
            try:
                self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
                self.gpt2_model.eval()
            except Exception:
                # disable if loading fails
                self.use_perplexity = False

    def fit(self, X, y=None):
        return self

    def _perplexity(self, text: str) -> float:
        if not self.use_perplexity or self.gpt2_model is None:
            return 0.0
        try:
            enc = self.gpt2_tokenizer(text, return_tensors="pt")
            import torch
            with torch.no_grad():
                outputs = self.gpt2_model(**enc, labels=enc["input_ids"])
                loss = outputs.loss.item()
            ppl = math.exp(loss)
            return float(ppl)
        except Exception:
            return 0.0

    def transform(self, X):
        rows = []
        for doc in X:
            if not isinstance(doc, str):
                doc = str(doc)
            chars = len(doc)
            words = re.findall(r"\w+", doc)
            word_count = len(words)
            avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0

            sentences = re.split(r"[.!?]+\s*", doc)
            sentences = [s for s in sentences if s.strip()]
            sent_count = len(sentences)
            sent_lens = [len(re.findall(r"\w+", s)) for s in sentences]
            avg_sent_len = (sum(sent_lens) / sent_count) if sent_count else 0.0
            std_sent_len = (np.std(sent_lens) if sent_count else 0.0)

            types = set(w.lower() for w in words)
            ttr = (len(types) / word_count) if word_count else 0.0

            # hapax legomena ratio
            freqs = {}
            for w in (w.lower() for w in words):
                freqs[w] = freqs.get(w, 0) + 1
            hapax = sum(1 for v in freqs.values() if v == 1)
            hapax_ratio = (hapax / word_count) if word_count else 0.0

            punct_count = len(re.findall(r"[\.,;:!?\\-]", doc))
            punct_ratio = (punct_count / chars) if chars else 0.0

            stop_count = sum(1 for w in (w.lower() for w in words) if w in self.stop_words)
            stop_ratio = (stop_count / word_count) if word_count else 0.0

            capital_count = sum(1 for w in words if any(c.isupper() for c in w))
            capital_ratio = (capital_count / word_count) if word_count else 0.0

            ppl = self._perplexity(doc)

            # burstiness: coefficient of variation of sentence lengths
            burstiness = (std_sent_len / (avg_sent_len + 1e-9)) if avg_sent_len else 0.0

            # Zipf slope: fit line on log(rank) vs log(freq) for frequent words
            zipf_slope = 0.0
            try:
                freqs_list = sorted(freqs.values(), reverse=True)
                if len(freqs_list) > 2:
                    # use top N ranks for stability
                    topn = min(50, len(freqs_list))
                    freqs_arr = np.array(freqs_list[:topn], dtype=float)
                    ranks = np.arange(1, freqs_arr.shape[0] + 1, dtype=float)
                    # avoid zeros
                    mask = freqs_arr > 0
                    if mask.sum() > 2:
                        log_r = np.log(ranks[mask])
                        log_f = np.log(freqs_arr[mask])
                        # slope of log(freq) = a + b*log(rank)
                        b = np.polyfit(log_r, log_f, 1)[0]
                        zipf_slope = float(b)
            except Exception:
                zipf_slope = 0.0

            rows.append([
                chars,
                word_count,
                avg_word_len,
                sent_count,
                avg_sent_len,
                std_sent_len,
                burstiness,
                ttr,
                hapax_ratio,
                punct_ratio,
                stop_ratio,
                capital_ratio,
                ppl,
                zipf_slope,
            ])

        return np.array(rows)


def build_and_train_model(seed: int = 42, force_retrain: bool = False, use_perplexity: bool = False, use_transformer: bool = False, language: str = "auto", use_hf: bool = False, hf_model_name: str = None):
    """Train and return pipeline, report, confusion matrix.

    Parameters
    - use_perplexity: include GPT-2 perplexity feature (requires transformers)
    - use_transformer: use sentence-transformers embeddings instead of TF-IDF
    - language: 'auto'|'en'|'zh' (handled below)
    """
    # use_hf: if True, use a HuggingFace fine-tuned `text-classification` model
    # specified by `hf_model_name` instead of training a local scikit-learn model.
    # The HF pipeline must be available (install `transformers`).
    # Small demo dataset: human vs AI examples. Extend as needed.
    human_texts = [
        "中文內容畢竟屬於中國文化，而不是台灣文化；台灣就像新加坡一樣，只是會中文的華人而已，本身沒有太多文化累積，一部分的人也並非華人。台灣唯一較具代表性的廟宇文化可能也不如中國傳統完整；至於原住民文化，中國可能也保留得更原始、更完整，而台灣的原住民文化多已被同化，我認識的原住民甚至很多都不會講原住民語。",
        "1880年美國總統選舉於11月2日舉行，是歷史上第24次美國總統大選。本次的競爭雙方主要是共和黨候選人詹姆士·艾布拉姆·加菲爾和民主黨候選人溫菲爾德·史考特·漢考克，最終加菲爾勝出當選美國總統，選民投票率在美國歷史上屬最高之列。兩位候選人的普選票總數差距尚不足2000，截至2024年大選，這仍然是美國所有總統大選中普選票差距最小的一次。",
        "法蘭克王國於6世紀至8世紀向萊茵河流域擴張，其中以查理曼時期最盛；800年，查理曼被教宗加冕為「羅馬皇帝」，成為476年西羅馬帝國滅亡後第一位此頭銜擁有者。查理曼的三個孫子於843年據《凡爾登條約》三分法蘭克，其中，東法蘭克被日耳曼人路易所統治。",
        "在光的傳播過程中，光線照射到粒子時，如果粒子大於入射光波長很多倍，則發生光的反射；如果粒子小於入射光波長，則發生光的散射，這時觀察到的是光波環繞微粒而向其四周放射的光，稱為散射光或乳光。廷得耳效應就是光的散射現象或乳光現象。由於溶膠粒子大小一般不超過100 nm，膠體粒子介於溶液中溶質粒子和濁液粒子之間，其大小在1～100nm。小於可見光波長（400 nm～700 nm），因此，當可見光透過溶膠時會產生明顯的散射作用。而對於真溶液，雖然分子或離子更小，但因散射光的強度隨散射粒子體積的減小而明顯減弱，因此，真溶液對光的散射作用很微弱。此外，散射光的強度還隨分散體系中粒子濃度增大而增強。所以說，膠體能有廷得耳現象，而溶液幾乎沒有，可以採用廷得耳現象來區分膠體和溶液。1869年，英國科學家約翰·廷得耳研究了此現象。",
        "克卜勒定律（英語：Kepler's law）是由德國天文學家、數學家約翰尼斯·克卜勒所發現的、關於行星運動的定律。克卜勒於1609年在他出版的《新天文學》科學雜誌上發表了關於行星運動的兩條定律，又於1618年，發現了第三條定律。",
        "欸，不得了啦！下週島語在桃園的新分店要開放訂位了，該不會又要拼手速了吧？看起來連平日都會很難訂到。",
        "常看到說要打擊詐騙，推出了很多措施，但最近發現，這些結果是人民辦個帳戶就被問東問西、問祖宗十八代，轉個帳就被凍結，領個錢還要掃臉，看個短片APP也直接被鎖，生活上出現許多匪夷所思的不方便；而真正的詐騙集團被抓到，卻往往小額交保、談笑風生回家開趴。這裡面有沒有什麼打詐的八卦呢？",
        "我記得石器時代是華義第二款網路遊戲，第一款好像叫做'人在江湖'。這遊戲我當時玩一玩只覺得哇靠，真是太特殊了，很多東西真的都很原始。記得印象最深的交易方式是雙方確認交易後要把東西丟在地上，之後還互相約定數到三，走到對方那邊去做交換動作（找沒人的空地方進行交易）。時常這樣交易會被隔空攔截，有一次我跟別人買光精項鍊這東西，那次當我把錢丟地上後，對方也把道具丟地上，結果我還沒撿到我要買的東西，那邊就被不知名人從陰影冒出搶走。不甘心的我就只好琁回去把自己錢給撿起來，幸好撿成功，不然可虧大。不過後來聽說這也是一種詐騙，搶東西跟賣你東西都是他們自己人，預謀好的，只要主動提出交易地點之後就是等誘拐你上當。",
        "這次 Google 退出中國，大家討論的是中國資訊封鎖的事情，但另一導火線是 Google 亦遭駭客攻擊，企圖奪取幾位人權份子的資訊。讓我囧的是，大部份攻擊 IP 來自台灣。於是 Google 了一下，台灣雖生產了世界七成的電腦及零組件，但一般民眾資訊網路安全概念實在是有夠差的。",
        "演算法，在數學（算學）和電腦科學中指一個被定義好的、電腦可施行其指示的有限步驟或次序，常用於計算、資料處理和自動推理。演算法可以使用條件語句通過各種途徑轉移代碼執行（稱為自動決策），並推導出有效的推論（稱為自動推理），最終實現自動化。"
    ]

    ai_texts = [
        "當然還有！如果你想打造一個「超完整的娛樂 Discord Bot」，下面是更多進階、特別、有創意的功能，你可以隨便挑，我都可以寫給你！",
        "今天路上遇到一位老人，他慢慢地牽著一隻小狗，看起來像在散步，也像在回憶過去的時光。看到他們的樣子，我突然覺得生活中最簡單的幸福，其實就是這樣。",
        "最近有點想換手機，但一直猶豫不決，朋友說我挑剔，其實只是想找一台不容易讓我後悔的，選擇困難症又發作了。",
        "前幾天朋友推薦我玩一款手遊，結果我一玩就停不下來，整整兩天幾乎沒睡好。雖然遊戲機制很有趣，但我開始有點對不起自己的作息，連早餐都忘了吃。看著手機裡幾百個未完成任務，心裡既焦慮又期待下一步挑戰。朋友半開玩笑說我沉迷了，但其實這也是一種短暫的放鬆吧，平常工作壓力大，能暫時專注在另一個世界裡，也算是一種逃避，但又不完全是壞事，至少心情上能有一點調劑。",
        "人工智慧（Artificial Intelligence, AI）是指模擬人類智能的計算系統或軟體，主要功能包括感知、推理、學習和決策。自20世紀中期誕生以來，人工智慧經歷了數次技術浪潮，包括專家系統、機器學習以及深度學習的快速發展。現今，人工智慧已廣泛應用於醫療診斷、金融分析、交通管理、語音與影像辨識等領域。人工智慧的主要技術基礎包括神經網路、自然語言處理（NLP）以及強化學習。其應用帶來效率提升和成本降低，但同時引發倫理、隱私及就業影響等問題。各國政府及國際組織已開始制定人工智慧治理框架，旨在確保技術安全、透明並兼顧社會責任。未來，人工智慧將可能與物聯網、雲計算及量子運算結合，進一步推動智能化社會的發展。",
        "最近天氣真的讓人受不了，早上出門冷得要命，中午又熱到想脫掉外套，傍晚一吹風又開始發抖。我每次出門都要穿三層衣服，但還是常常冷到打哆嗦，感覺自己完全被天氣操控。朋友說這叫季節性憂鬱，我倒是覺得只是懶得適應而已。每天上下班都得這樣穿脫衣服，光是想到就覺得累，但偶爾看到陽光透過樹葉灑在路上，又覺得生活還是有點小確幸。",
        "FCN 和 U-Net 在融合編碼器的淺層特徵與解碼器的深層特徵時，使用的方法有所不同：FCN 通常是直接將編碼器最後的特徵圖（深層特徵）進行上採樣，並通過卷積層進行處理。它沒有專門設計的機制來直接融合淺層特徵和深層特徵。淺層特徵通常不會直接與深層特徵進行合併。U-Net 使用了「跳躍連接（Skip Connections）」的方式，將編碼器的淺層特徵直接與解碼器的深層特徵進行融合。在解碼器的每一層，上採樣後會與對應的編碼器層的特徵圖進行拼接，這樣可以保留更多細節信息，有助於提高分割結果的精確度。",
        "上週末去了一家新開的拉麵店，湯頭真的意外地濃郁，但麵稍微有點軟，不過整體還是值得再去一次。",
        "社交媒體指的是透過網路平台進行社交互動和內容分享的工具，包括 Facebook、Twitter、Instagram 等。自2000年代初期興起以來，社交媒體已成為全球資訊傳播和人際互動的重要渠道。其主要特徵包括即時性、互動性以及內容生成的多樣化，使個人和企業能以低成本傳播信息。社交媒體在社會、政治和經濟方面均有深遠影響。例如，它促進了公民參與、品牌行銷和資訊透明，但也帶來隱私洩露、網絡霸凌和虛假信息傳播的問題。各國政府和平台公司針對此類問題制定政策，包括數據保護規範、內容審查和演算法透明化等措施。研究指出，社交媒體的使用與心理健康呈現複雜關聯，其正負面影響取決於使用方式、使用時長及個人社交資源等因素。",
        "作為一種多任務訓練方法，確定引入額外任務對模型在原始任務上表現的影響是至關重要的。為了驗證序列標註模塊對對應結果的影響，我們進行了一個消融實驗，對比有無序列標註模塊的情況。如表7所示，對應模塊在有序列標註模塊的情況下，在列和值的對齊上確實達到了更好的性能。通過分析，我們發現性能提升主要來自於長概念的提及（標記長度 > 4）。原因是序列標註模塊中的 CRF 層可以增強對應模塊捕捉長距離依賴的能力。總結來說，我們可以得出結論：序列標註任務可以與對應任務良好配合。"
    ]

    # Augment training data to include longer human examples to reduce length bias
    random.seed(seed)
    long_humans = []
    for _ in range(50):
        k = random.randint(2, 6)
        long_humans.append("\n\n".join(random.choices(human_texts, k=k)))

    # Also add some longer AI examples to cover length variety
    long_ais = []
    for _ in range(30):
        k = random.randint(2, 6)
        long_ais.append("\n\n".join(random.choices(ai_texts, k=k)))

    texts = human_texts + ai_texts + long_humans + long_ais
    labels = [0] * (len(human_texts) + len(long_humans)) + [1] * (len(ai_texts) + len(long_ais))

    # model persistence path (same folder as this script)
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "ai_detector_model.joblib")

    # Note: optionally a user can pass `use_hf=True` and `hf_model_name="<hf-model>"`
    # to use a HuggingFace fine-tuned text-classification model. We detect HF
    # pipeline availability above; if not installed we will fall back to the
    # classic scikit-learn pipeline.

    if os.path.exists(model_path) and not force_retrain:
        try:
            data = load(model_path)
            saved_use_ppl = data.get("use_perplexity", False)
            saved_use_transformer = data.get("use_transformer", False)
            saved_lang = data.get("language", "auto")
            saved_use_hf = data.get("use_hf", False)
            saved_hf_name = data.get("hf_model_name", None)
            # If requested feature flags changed, force retrain
            if saved_use_ppl != use_perplexity or saved_use_transformer != use_transformer or saved_lang != language or saved_use_hf != use_hf or saved_hf_name != hf_model_name:
                pass
            else:
                pipeline = data.get("pipeline")
                report = data.get("report")
                cm = data.get("cm")
                return pipeline, report, cm
        except Exception:
            # fall through to retrain on load error
            pass

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # If user requested a HuggingFace fine-tuned classifier, try to load it
    if use_hf:
        if not _HAS_HF_PIPELINE:
            # fall back to local pipeline if HF pipeline is not installed
            print("transformers pipeline not available; falling back to local pipeline")
        else:
            # require a model name
            if not hf_model_name:
                raise ValueError("hf_model_name must be provided when use_hf=True")
            try:
                hf = hf_pipeline("text-classification", model=hf_model_name, return_all_scores=True)
            except Exception as e:
                print("Failed to load HF model:", e)
                hf = None

            if hf is not None:
                # evaluate the HF model on the test split
                def _hf_extract_ai_prob(result_scores):
                    # result_scores may be a list of dicts or list-of-lists depending on call
                    entries = None
                    if isinstance(result_scores, list) and len(result_scores) > 0 and isinstance(result_scores[0], list):
                        entries = result_scores[0]
                    elif isinstance(result_scores, list) and len(result_scores) > 0 and isinstance(result_scores[0], dict):
                        entries = result_scores
                    else:
                        entries = list(result_scores)

                    ai_score = None
                    human_score = None
                    for e in entries:
                        lbl = str(e.get("label", "")).lower()
                        sc = float(e.get("score", 0.0))
                        if any(k in lbl for k in ("ai", "machine", "bot", "generated", "gpt", "synthetic", "model")):
                            ai_score = sc
                        if any(k in lbl for k in ("human", "real", "people")):
                            human_score = sc

                    # fallback heuristics
                    if ai_score is None and human_score is None:
                        if len(entries) >= 2:
                            # assume second label corresponds to AI
                            ai_score = float(entries[1].get("score", 0.0))
                            human_score = float(entries[0].get("score", 0.0))
                        else:
                            ai_score = float(entries[0].get("score", 0.0))
                            human_score = 1.0 - ai_score
                    elif ai_score is None:
                        ai_score = 1.0 - human_score
                    elif human_score is None:
                        human_score = 1.0 - ai_score

                    return ai_score, human_score

                probs = []
                preds = []
                for t in X_test:
                    try:
                        out = hf(t, return_all_scores=True)
                        ai_sc, human_sc = _hf_extract_ai_prob(out)
                        probs.append([human_sc, ai_sc])
                        preds.append(1 if ai_sc >= 0.5 else 0)
                    except Exception:
                        probs.append([0.5, 0.5])
                        preds.append(0)

                # build report and confusion matrix using preds
                report = metrics.classification_report(y_test, preds, output_dict=True)
                cm = metrics.confusion_matrix(y_test, preds)

                # save HF pipeline + metadata
                try:
                    dump({
                        "pipeline": hf,
                        "report": report,
                        "cm": cm,
                        "use_perplexity": use_perplexity,
                        "use_transformer": use_transformer,
                        "language": language,
                        "use_hf": True,
                        "hf_model_name": hf_model_name,
                    }, model_path)
                except Exception:
                    pass

                return hf, report, cm

    # create a combined pipeline: TF-IDF (or transformer embed) + stylometric features
    stats = TextStats(use_perplexity=use_perplexity)
    # Configure TF-IDF for Chinese vs others
    def contains_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    use_chinese = False
    if language == "zh":
        use_chinese = True
    elif language == "auto":
        sample_text = " ".join(texts[:5])
        use_chinese = contains_cjk(sample_text)

    # Choose TF-IDF tokenizer/settings
    if use_chinese:
        if _HAS_JIEBA:
            tfidf = TfidfVectorizer(tokenizer=jieba.lcut, analyzer="word", ngram_range=(1, 2), max_df=0.9, min_df=1)
        else:
            tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_df=0.95)
    else:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=1)

    # FeatureUnion expects transformer objects; stats produces dense features
    if use_transformer and _HAS_SB:
        # choose multilingual model for Chinese
        model_name = "paraphrase-multilingual-MiniLM-L12-v2" if use_chinese else "all-MiniLM-L6-v2"
        embedder = TransformerEmbedder(model_name)
        emb_stats = EmbeddingStats(model_name)
        union = FeatureUnion([
            ("embed", make_pipeline(embedder)),
            ("embed_stats", make_pipeline(emb_stats, StandardScaler())),
            ("stats", make_pipeline(stats, StandardScaler())),
        ])
    else:
        # include embedding stats even if sentence-transformers not available
        emb_stats = EmbeddingStats()
        union = FeatureUnion([
            ("tfidf", tfidf),
            ("embed_stats", make_pipeline(emb_stats, StandardScaler())),
            ("stats", make_pipeline(stats, StandardScaler())),
        ])

    # after FeatureUnion the result may be sparse; convert to dense for the classifier
    to_dense = FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X)

    pipeline = make_pipeline(
        union,
        to_dense,
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed, class_weight="balanced"),
    )

    pipeline.fit(X_train, y_train)

    # Evaluate on test split for optional display
    preds = pipeline.predict(X_test)
    report = metrics.classification_report(y_test, preds, output_dict=True)
    cm = metrics.confusion_matrix(y_test, preds)

    # save model + metadata (include flag whether perplexity was used)
    try:
        dump({
            "pipeline": pipeline,
            "report": report,
            "cm": cm,
            "use_perplexity": use_perplexity,
            "use_transformer": use_transformer,
            "language": language,
        }, model_path)
    except Exception:
        pass

    return pipeline, report, cm


def predict_text(model, text: str):
    # scikit-learn pipeline
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        ai_prob = float(proba[1])
        human_prob = float(proba[0])
        return human_prob, ai_prob

    # HuggingFace pipeline (heuristic extractor)
    def _hf_extract_ai_prob(result_scores):
        if isinstance(result_scores, list) and len(result_scores) > 0 and isinstance(result_scores[0], list):
            entries = result_scores[0]
        elif isinstance(result_scores, list) and len(result_scores) > 0 and isinstance(result_scores[0], dict):
            entries = result_scores
        else:
            try:
                entries = list(result_scores)
            except Exception:
                entries = []

        ai_score = None
        human_score = None
        for e in entries:
            lbl = str(e.get("label", "")).lower()
            sc = float(e.get("score", 0.0))
            if any(k in lbl for k in ("ai", "machine", "bot", "generated", "gpt", "synthetic", "model")):
                ai_score = sc
            if any(k in lbl for k in ("human", "real", "people")):
                human_score = sc

        if ai_score is None and human_score is None:
            if len(entries) >= 2:
                ai_score = float(entries[1].get("score", 0.0))
                human_score = float(entries[0].get("score", 0.0))
            elif len(entries) == 1:
                ai_score = float(entries[0].get("score", 0.0))
                human_score = 1.0 - ai_score
            else:
                ai_score = 0.5
                human_score = 0.5
        elif ai_score is None:
            ai_score = 1.0 - human_score
        elif human_score is None:
            human_score = 1.0 - ai_score

        return ai_score, human_score

    if _HAS_HF_PIPELINE and callable(model):
        try:
            out = model(text, return_all_scores=True)
            ai_sc, human_sc = _hf_extract_ai_prob(out)
            return human_sc, ai_sc
        except Exception:
            pass

    # fallback: uniform
    return 0.5, 0.5


def plot_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "AI"])
    ax.set_yticklabels(["Human", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    return fig
