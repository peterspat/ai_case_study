"""
Module Name: Page Markdown Template
Author: Kenneth Leung
Last Modified: 19 Mar 2023
"""

home_page = """
# Home

Welcome to the NLP Blog Analysis Demo! This website showcases the results of an automated analysis pipeline designed to analyze blog entries. Here's a brief overview of what you'll find:

## Data Exploration
Begin with exploring the dataset to gain insights into its structure and characteristics. Conduct basic statistical analysis and visualize the data to understand its distribution and properties.

## Data Preprocessing Pipeline
In this phase, demonstrate the preprocessing pipeline, highlighting each step along with its intermediate results. Tasks include data cleaning, tokenization, and vectorization.

## Final Analysis
Conclude with the analysis post preprocessing. Showcase the results of classification and clustering, and employ visualization techniques such as tag clouds to present the findings effectively.

Feel free to explore the demo and interact with the presented results. 

<object data="src/assets/presentation.pdf" type="application/pdf" class="responsive">
    <embed src="src/asset/presentation.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="src/assets/presentation.pdf">Download PDF</a>.</p>
    </embed>
</object>
<a href="src/assets/presentation.pdf" download>Download PDF Presentation</a>

<style>
.responsive {
  width: 700px;
  height: 400px;
}
</style>

"""