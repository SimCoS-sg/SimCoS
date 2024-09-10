# Simple Contrastive Learning with Knowledge Graphs for Story Generation
## Appendix
Table 1 presents the generated stories in OutGen from $\text{LongLM}_{\text{base}}$ and SimCoS. $\text{LongLM}_{\text{base}}$ incorporating multiple phrases struggles with logical consistency and verbosity, often repeating themes of solitude. In contrast, SimCoS utilizes a greater number of phrases more effectively, creating a coherent and logically structured narrative that included a variety of elements such as the blooming process, human interactions, and a final ending.
![case study](./fig.pdf)

Fig. 3 presents an analysis of performance variations with different $\alpha$ values on the OutGen test set. This hyperparameter impacts the measurement of distances between story subgraphs, with extreme values potentially skewing towards either text or logical similarity. This underscores the sensitivity of Distinct and Coverage metrics to $\alpha$ selection, emphasizing the importance of balance to ensure comprehensive evaluation in story generation.
![performance variations with different $\alpha$ values on the OutGen test set](./fig3.pdf)
