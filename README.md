# Demand Forecast Analysis

## A Case Study in Demand Forecasting for Supply Chain Optimization

### Introduction

This case study delves into the application of demand forecasting to address issues plaguing the supply chain of an e-commerce company. The project aims to elucidate the primary objectives, the systematic approach taken to identify the root cause of the problem, and the subsequent development of a demand forecasting model designed to alleviate the issues at hand.

### Discovery

#### Problem

The e-commerce company in question was inundated with customer complaints related to undelivered or perpetually unavailable products, causing substantial disruptions to the business. 

#### Interviews

Various stakeholders were interviewed to comprehend the nature of the problem:

- **Customers:** Customers reported not receiving products they had ordered.
- **Business Department:** They posited that the problem was systemic and struggled to quantify the extent of undelivered products.
- **Technology Department:** Their investigations revealed that integration issues had been rectified, but orders were not being transmitted to the logistics team.
- **Logistics Department:** The logistics team claimed they had the necessary resources for deliveries but faced difficulties due to product unavailability.

#### Root Causes

The investigation revealed two root causes for the issues:

- **Listing Out-of-Stock Products:** Products listed as available on the e-commerce platform were, in fact, out of stock.
- **Lack of Predictability:** The company lacked an effective system to predict which products required more stock, contributing to the problem's persistence.

#### Solutions

To address these issues, the following solutions were proposed:

- **Systemic Issue Resolution:** Rectify the systemic issue of listing products as available when they are out of stock and provide improved visibility to the business and logistics teams.
- **Demand-Based Stock Optimization:** Optimize stock levels based on demand patterns to mitigate the root cause of the problem.

#### Benefits

Implementing these solutions is expected to yield several advantages, including:

- **Revenue Increase:** A reduction in the number of canceled purchases due to unavailability will lead to increased revenue.
- **Operational Optimization:** Improved metrics for the logistics and business teams, leading to more efficient operations.

### Data Understanding

For this project, data from the Kaggle dataset "Sample Sales Data" (available at [link](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)) was employed. Before diving into analysis, it is essential to gain a comprehensive understanding of the dataset and establish connections between the data variables and business context.

#### Mind Map

To facilitate this understanding, a mind map was constructed, providing an overview of the dataset and its relevance to the business problem:

![mind_map.png](mind_map.png)

### Generating Hypotheses

After gaining a firm understanding of the data, the next step involves crafting hypotheses. These hypotheses serve a dual purpose: connecting the data analysis to the business problem and providing valuable insights to guide the decision-making process.

Hypotheses are a cornerstone of our analytical approach, not only for building predictive models using machine learning algorithms but also for addressing various business queries and uncertainties. By formulating and testing hypotheses, we ensure that our analytical efforts yield value beyond predictive modeling. This approach safeguards against the risk of the model failing to make accurate predictions, as it provides additional insights and aids in informed decision-making based on the analysis of each hypothesis.