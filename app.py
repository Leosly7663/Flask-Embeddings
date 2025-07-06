from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import re
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')

def split_sentences(text):
    raw = re.split(r'[.\n]+', text)
    return [s.strip() for s in raw if s.strip()]

def combine_sentences(sentences):
    combinations = []
    for i, s in enumerate(sentences):
        combinations.append(s)  # Single sentence
        if i < len(sentences) - 1:
            combined = f"{s}. {sentences[i+1]}"
            combinations.append(combined)
    return combinations

@app.route('/combine-match', methods=['POST'])
def combine_similarity():
    data = request.get_json()
    resume_text = data.get('resume')
    job_sentences = data.get('jobSentences')

    if not resume_text or not job_sentences:
        return jsonify({'error': 'resume and jobSentences are required'}), 400

    # Split and combine sentences
    base_sentences = split_sentences(resume_text)
    combined_sentences = combine_sentences(base_sentences)

    # Embed all combinations once
    combined_embeddings = model.encode(combined_sentences, convert_to_tensor=True)
    job_embeddings = model.encode(job_sentences, convert_to_tensor=True)

    results = []

    # For each job sentence
    for i, job_sentence in enumerate(job_sentences):
        job_embedding = job_embeddings[i]
        similarities = util.cos_sim(job_embedding, combined_embeddings)[0]

        # Find the best matching combined sentence
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        best_sentence = combined_sentences[best_idx]

        results.append({
            "jobSentence": job_sentence,
            "bestMatchSentence": best_sentence,
            "similarityPercent": float(round(best_score * 100, 2))
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
