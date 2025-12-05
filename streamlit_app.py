import streamlit as st
import pandas as pd
import numpy as np
from ai_model import build_and_train_model, predict_text, plot_confusion_matrix, _HAS_TRANSFORMERS, _HAS_JIEBA, _HAS_SB


def main():
	st.title("AI vs Human Text Detector — Demo")

	st.markdown(
		"Paste a piece of text below and the app will show a simple AI/Human probability estimate."
	)

	# Language selection (Auto / English / Chinese)
	language_choice = st.selectbox("Input language", ["Auto", "English", "Chinese"], index=0)
	language = "auto" if language_choice == "Auto" else ("zh" if language_choice == "Chinese" else "en")

	# Load model (or train if missing). Use persisted model when available.
	model, report, cm = build_and_train_model(use_perplexity=False, use_transformer=False, language=language)

	# bind the text area to session state so buttons can update it without rerun
	text_input = st.text_area("Input text", height=200, key="input_text")

	col1, col2 = st.columns([3, 1])

	# Keep the session state text in sync (text_area is keyed to session_state)
	text_input = st.session_state.get("input_text", "")

	# Send button: user clicks to submit text for prediction
	with col2:
		send = st.button("Send")

	if send:
		if text_input and text_input.strip():
			human_p, ai_p = predict_text(model, text_input)
			st.session_state["last_result"] = {"human": human_p, "ai": ai_p}
		else:
			st.warning("Please enter text before sending.")

	# Show latest result if available
	last = st.session_state.get("last_result")
	if last:
		human_p = last.get("human", 0.5)
		ai_p = last.get("ai", 0.5)

		ai_pct = round(ai_p * 100, 2)
		human_pct = round(human_p * 100, 2)

		st.subheader("Prediction")
		st.write(f"**AI:** {ai_pct}% — **Human:** {human_pct}%")

		st.progress(int(ai_pct))

		# Optional: show probability meter-style bars
		st.write("---")
		st.write("**Probability distribution**")
		df = pd.DataFrame({"Human": [human_p], "AI": [ai_p]})
		st.bar_chart(df)
	else:
		st.info("Enter some text and click 'Send' to get a prediction.")

	# Metrics & visualization (optional)
	with st.expander("Model evaluation (demo)"):
		st.write("Classification report on a small demo holdout set:")
		st.json(report)
		st.write("Confusion matrix:")
		fig = plot_confusion_matrix(cm)
		st.pyplot(fig)


if __name__ == "__main__":
	main()

