from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import json
from datetime import datetime


app = Flask(__name__)
CORS(app)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)

def init_db():
    conn = sqlite3.connect('pages.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS pages
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         url TEXT UNIQUE,
         title TEXT,
         content TEXT,
         embedding BLOB,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()

@app.route('/process', methods=['POST'])
def process_page():
    data = request.json
    
    try:
        # Generate embedding for the content
        embedding = model.encode(data['content'])
        
        # Store in SQLite
        conn = sqlite3.connect('pages.db')
        c = conn.cursor()
        
        # Insert and get the ID
        c.execute('''
            INSERT OR REPLACE INTO pages (url, title, content, embedding)
            VALUES (?, ?, ?, ?)
        ''', (data['url'], data['title'], data['content'], embedding.tobytes()))
        
        # Get the ID of the inserted row
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Add to FAISS index with the correct index
        if row_id is not None:
            index.add(embedding.reshape(1, -1))
        
        return jsonify({'status': 'success', 'id': row_id})
        
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    k = request.json.get('k', 5)
    
    try:
        # Generate query embedding
        query_embedding = model.encode(query)
        
        # Search in FAISS
        D, I = index.search(query_embedding.reshape(1, -1), k)
        
        # Get results from SQLite
        conn = sqlite3.connect('pages.db')
        c = conn.cursor()
        results = []
        
        for i, idx in enumerate(I[0]):
            # Convert numpy.int64 to regular integer
            idx_int = int(idx)  # Convert FAISS index to Python int
            
            try:
                c.execute('SELECT url, title, content FROM pages WHERE id = ?', (idx_int + 1,))
                row = c.fetchone()
                if row:
                    url, title, content = row
                    results.append({
                        'url': url,
                        'title': title,
                        'content': content[:200] + "...",  # Truncate content for preview
                        'score': float(D[0][i])  # Convert numpy.float32 to Python float
                    })
            except sqlite3.Error as e:
                print(f"SQLite error for index {idx_int}: {e}")
                continue
        
        conn.close()
        return jsonify(results)
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)