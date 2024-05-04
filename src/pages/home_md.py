"""
Module Name: Page Markdown Template
Author: Kenneth Leung
Last Modified: 19 Mar 2023
"""

home_page = """
# Home

Welcome to the NLP Blog Analysis Demo! This website showcases the results of an automated analysis pipeline designed to analyze blog entries. Here's a brief overview of what you'll find:

## Data Preprocessing
The first step involves preparing and cleaning the dataset. This includes tasks such as classifying quotes, tokenization, general cleaning, and vectorization.

## Classification
Next, the texts are categorized, for example, into sentiments and overarching themes. 

## Clustering
Individual texts or topics are grouped based on similarities. We use k-Means Clustering, a common algorithm for this task.

## Result Presentation
Finally, the results from the classification and clustering steps are presented. You'll find various visualization techniques such as tag clouds to help understand the analyzed data.

Feel free to explore the demo and interact with the presented results. 

<object data="assets/presentation.pdf" type="application/pdf" class="responsive">
    <embed src="asset/presentation.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="assets/presentation.pdf">Download PDF</a>.</p>
    </embed>
</object>

<style>
.responsive {
  width: 700px;
  height: 400px;
}
</style>

"""