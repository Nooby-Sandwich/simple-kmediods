from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from kmedoids import k_medoids

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    # Parse form data
    num_clusters = int(request.form.get('num_clusters'))
    num_points = int(request.form.get('num_points'))

    # Generate random data
    X = np.random.rand(num_points, 2)

    # Apply K-Medoids clustering
    medoids, labels = k_medoids(X, num_clusters)

    # Plot the clusters
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for i in range(num_clusters):
        points = X[labels == i]
        ax.scatter(points[:, 0], points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')
        ax.scatter(medoids[i][0], medoids[i][1], c='black', marker='x', s=100)  # Medoid
    
    ax.set_title(f'K-Medoids Clustering with k={num_clusters}')
    ax.legend()

    # Save the plot to a PNG image in memory
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
