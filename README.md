# CVSS Base Score Prediction Using an Optimized Machine Learning Scheme
## Institute of Electrical and Electronics Engineers (IEEE) · Aug 25, 2023

### Conducted and led NSF-sponsored research advised by Prof. Qinghua Li at UARK’s SEEDS Lab investigating the use of language learning models and machine learning algorithms (Doc2Vec, Fast.ai, Advanced BERT Models) to improve the Common Vulnerability Scoring System and effectively classify computer system vulnerabilities. Accepted for publication by TechConnect for Resilience Week 2023 in Washington D.C.

## Abstract:
The Common Vulnerability Scoring System (CVSS) is commonly used to measure the severity of software vulner-abilities. It consists of a CVSS Score Vector (i.e., a vector of metrics) and a CVSS Base Score calculated based on the vector. The Base Score is widely used by electric utilities to measure the risk levels of vulnerabilities and prioritize remediation actions. However, the process of determining the CVSS metric values is currently very time-consuming since it is manually done by human experts based on text descriptions of vulnerabilities, which increases the delays of remediating vulnerabilities and hence increases security risks at electric utilities. In this paper, we develop an efficient and effective solution to automatically predict the CVSS Base Score of vulnerabilities primarily based on their text descriptions, leveraging Natural Language Processing and machine learning techniques. Text descriptions for tens of thousands of vulnerabilities are comprehensively interpreted and vectorized using Doc2Vec, fed to a neural network with a condensed regression structure, which is then fine-tuned using Bayesian Optimization. By exploring and selecting the most efficient option at each stage of development, we create an optimized scheme that predicts CVSS Base Scores with very low error. Our work shows that it is possible to effectively predict CVSS Base Scores using simple but optimized neural networks. It makes crucial progress toward addressing the inefficiencies of the current CVSS severity assessment process through automation.

## Running this program
Run the following command in the terminal:

```
  python main.py
```
