.. _datasets:

Datasets
*********************************************

Overview of the currently available datasets in MAMKit.

.. list-table:: **Datasets Information**
   :header-rows: 1
   :widths: 20 15 15 50

   * - **Datasets**
     - **Tasks**
     - **Modalities**
     - **Description**
   * - **UKDebates**
     - ASD
     - Text, Audio
     - The first MAM dataset. It contains transcriptions and audio sequences of three candidates for UK Prime Ministerial elections of 2015 in a two-hour debate aired by Sky News. The candidates are David Cameron, Nick Clegg, and Ed Miliband. The dataset contains 386 sentences and corresponding audio samples.
   * - **MArg** :sub:`γ`
     - ARC
     - Text, Audio
     - A multimodal dataset built around the 2020 US Presidential elections. The dataset contains transcriptions and audio sequences of four candidates and a debate moderator concerning 18 topics. The authors design a controlled crowdsourcing data annotation process whereby each crowd worker labels sentence pairs as describing support, attack, or no relation. In total, the dataset contains 4,104 sentence pairs with corresponding aligned audio samples. A high-quality subset of the M-Arg, M-Arg :sub:`γ` , containing 2,443 sentence pairs with high agreement confidence γ ≥ 85% is commonly considered for model evaluation.
   * - **MM-USED**
     - ASD, ACC
     - Text, Audio
     - It contains presidential candidates’ debate transcripts and corresponding audio recordings aired from 1960 to 2016. This dataset consists of 23,505 labeled sentences[1]_  and corresponding audio samples covering 39 debates and 26 different speakers.
   * - **MM-USED-fallacy**
     - AFC, AFD
     - Text, Audio
     - The dataset contains 1,278 samples[1]_ labeled as argumentative fallacies belonging to six distinct categories. Sentences are taken from presidential candidates’ debates aired from 1960 to 2016.


[1] Note: the dataset has undergone a refinement in the alignment process, which has resulted in adjustments to the number of samples included compared to the original versions published in the referenced papers.
