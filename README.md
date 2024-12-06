Team Contributions

Our project is a collaborative effort, with each team member focusing on specific components to improve and extend the baseline NGCF model. Below are the detailed contributions:

Enis Ozden

Loss Function Enhancements:

Designed and implemented Adaptive Bayesian Personalized Ranking (ABPR), a modified loss mechanism to prioritize harder-to-rank user-item pairs and improve subtle distinctions in ranking quality.

Integrated hinge loss, introducing margin constraints to separate relevant and irrelevant items more distinctly in the latent space.

Conducted extensive experiments to compare these loss mechanisms with the baseline Bayesian Personalized Ranking (BPR) loss, analyzing their impact on metrics such as Recall@K, Precision@K, and NDCG@K.


Optimization and Tuning:

Optimized the hyperparameters of the loss mechanisms to ensure robust training and better performance on sparse datasets like Gowalla.

Debugged issues related to loss function stability during the training phase and refined the implementation for compatibility with the NGCF framework.


Documentation and Analysis:

Authored detailed sections of the methodology and results analysis in the report, focusing on the impact of enhanced loss mechanisms.



Teoman Kaman

Regularization Techniques:

Researched, implemented, and integrated Layer Normalization into the NGCF pipeline to stabilize the training process and maintain consistent embedding distributions across layers.

Experimented with DropEdge, a graph regularization technique, to reduce graph complexity during training, ensuring better generalization and computational efficiency.


Efficiency Improvements:

Focused on reducing computational overhead by tuning regularization methods, enabling the model to achieve high performance with fewer compute units and reduced training time.

Evaluated the trade-offs between computational cost and recommendation quality for real-world applicability.


Experimental Pipeline:

Managed the training pipeline, ensuring compatibility between loss mechanisms and regularization techniques.

Verified performance consistency by running multiple trials with fixed seeds to confirm results.


Reporting and Presentation:

Authored key sections of the report related to experimental setup, findings on regularization techniques, and their implications for large-scale deployments.
