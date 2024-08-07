.. _leaderboard:

Leaderboard
*********************************************
Tasks:
   - **ASD**: Argumentative Stance Detection
   - **ARC**: Argumentative Relation Classification
   - **ACC**: Argumentative Component Classification
   - **AFC**: Argumentative Fallacy Classification



.. list-table:: **Test classification performance on MAM datasets**
   :header-rows: 1
   :widths: 15 15 15 15 15 15

   * - **Model**
     - **UKDebates**
     - **M-Arg** :sub:`γ`
     - **MM-USED**
     - **MM-USED**
     - **MMUSED-fallacy**
   * - **Task**
     - **ASD**
     - **ARC**
     - **ASD**
     - **ACC**
     - **AFC**

   * - **Text Only**
     -
     -
     -
     -
     -
   * - BiLSTM
     - .552\ :sub:`± .047`
     - .120\ :sub:`± .006`
     - .811\ :sub:`± .004`
     - .663\ :sub:`± .002`
     - .525\ :sub:`± .113`
   * - BERT
     - .654\ :sub:`± .003`
     - .132\ :sub:`± .004`
     - .824\ :sub:`± .009`
     - .679\ :sub:`± .004`
     - .594\ :sub:`± .122`
   * - RoBERTa
     - .692\ :sub:`± .005`
     - .172\ :sub:`± .015`
     - .839\ :sub:`± .010`
     - .680\ :sub:`± .001`
     - .615\ :sub:`± .097`
   * - **Audio Only**
     -
     -
     -
     -
     -
   * - BiLSTM w/ MFCCs
     - .302\ :sub:`± .047`
     - .003\ :sub:`± .005`
     - .722\ :sub:`± .027`
     - .527\ :sub:`± .004`
     - **.657**  :sub:`± .000`
   * - BiLSTM w/ Wav2Vec2
     - .376\ :sub:`± .023`
     - .000\ :sub:`± .000`
     - **.774** :sub:`± .008`
     - **.596** :sub:`± .005`
     - .655\ :sub:`± .117`
   * - BiLSTM w/ HuBERT
     - .364\ :sub:`± .012`
     - **.024** :sub:`± .012`
     - .745\ :sub:`± .009`
     - .566\ :sub:`± .007`
     - .638\ :sub:`± .000`
   * - BiLSTM w/ WavLM
     - **.393** :sub:`± .040`
     - .010\ :sub:`± .010`
     - .772\ :sub:`± .015`
     - .583\ :sub:`± .002`
     - .652\ :sub:`± .000`
   * - Transformer w/ Wav2Vec2
     - .440\ :sub:`± .030`
     - .000\ :sub:`± .000`
     - **.771** :sub:`± .019`
     - .514\ :sub:`± .000`
     - .567\ :sub:`± .225`
   * - Transformer w/ HuBERT
     - .425\ :sub:`± .033`
     - .000\ :sub:`± .000`
     - .765\ :sub:`± .016`
     - .524\ :sub:`± .004`
     - **.629** :sub:`± .162`
   * - Transformer w/ WavLM
     - **.455** :sub:`± .004`
     - .000\ :sub:`± .000`
     - .768\ :sub:`± .005`
     - **.526** :sub:`± .004`
     - .594\ :sub:`± .217`
   * - **Text Audio**
     -
     -
     -
     -
     -
   * - BiLSTM w/ MFCCs
     - .528\ :sub:`± .039`
     - .065\ :sub:`± .014`
     - .807\ :sub:`± .010`
     - .662\ :sub:`± .006`
     - **.572** :sub:`± .099`
   * - BiLSTM w/ Wav2Vec2
     - **.533** :sub:`± .009`
     - .079\ :sub:`± .014`
     - .808\ :sub:`± .012`
     - .665\ :sub:`± .004`
     - .505\ :sub:`± .168`
   * - BiLSTM w/ HuBERT
     - .409\ :sub:`± .017`
     - .055\ :sub:`± .020`
     - .807\ :sub:`± .013`
     - .653\ :sub:`± .003`
     - .456\ :sub:`± .131`
   * - BiLSTM w/ WavLM
     - .501\ :sub:`± .022`
     - **.084** :sub:`± .016`
     - **.815** :sub:`± .006`
     - **.667** :sub:`± .000`
     - .526\ :sub:`± .174`
   * - MM-BERT w/ Wav2Vec2
     - **662** :sub:`± .004`
     - .153\ :sub:`± .017`
     - **841** :sub:`± .005`
     - .677\ :sub:`± .003`
     - .561\ :sub:`± .114`
   * - MM-BERT w/ HuBERT
     - .626\ :sub:`± .003`
     - **.160** :sub:`± .015`
     - .840\ :sub:`± .006`
     - .677\ :sub:`± .004`
     - **.599** :sub:`± .128`
   * - MM-BERT w/ WavLM
     - .654\ :sub:`± .019`
     - .152\ :sub:`± .008`
     - .836\ :sub:`± .005`
     - **.680** :sub:`± .004`
     - .580\ :sub:`± .103`
   * - MM-RoBERTa w/ Wav2Vec2
     - .674\ :sub:`± .009`
     - **.178** :sub:`± .012`
     - .833\ :sub:`± .006`
     - **.678** :sub:`± .003`
     - .608\ :sub:`± .126`
   * - MM-RoBERTa w/ HuBERT
     - .624\ :sub:`± .015`
     - .147\ :sub:`± .004`
     - .837\ :sub:`± .003`
     - .677\ :sub:`± .008`
     - .576\ :sub:`± .097`
   * - MM-RoBERTa w/ WavLM
     - **.687** :sub:`± .010`
     - .165\ :sub:`± .018`
     - **.837** :sub:`± .009`
     - **678** :sub:`± .003`
     - **.624** :sub:`± .074`
   * - CSA w/ Wav2Vec2
     - **.663** :sub:`± .014`
     - .137\ :sub:`± .027`
     - .822\ :sub:`± .002`
     - **.693** :sub:`± .001`
     - .555\ :sub:`± .118`
   * - CSA w/ HuBERT
     - .632\ :sub:`± .018`
     - **.160** :sub:`± .015`
     - .813\ :sub:`± .004`
     - **.693** :sub:`± .001`
     - **.582** :sub:`± .114`
   * - CSA w/ WavLM
     - .655\ :sub:`± .029`
     - .155\ :sub:`± .030`
     - **.833** :sub:`± .011`
     - .697\ :sub:`± .001`
     - .535\ :sub:`± .102`
   * - Ensemble w/ Wav2Vec2
     - **.586** :sub:`± .015`
     - **.011** :sub:`± .011`
     - .825\ :sub:`± .004`
     - **.681** :sub:`± .002`
     - **.612** :sub:`± .134`
   * - Ensemble w/ HuBERT
     - .531\ :sub:`± .039`
     - .010\ :sub:`± .004`
     - .822\ :sub:`± .007`
     - .681\ :sub:`± .003`
     - .611\ :sub:`± .107



