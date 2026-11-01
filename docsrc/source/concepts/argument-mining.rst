Multimodal argument mining
==========================

Argument mining identifies arguments in text and classifies their parts and relations.
Multimodal argument mining adds the speech the text was transcribed from.
In a debate, the tone and the prosody of a speaker can carry information that the transcript loses.
MAMKit therefore treats every sample as a sentence that has a transcript, an audio clip, or both.

Tasks
-----

MAMKit covers five classification tasks.
Each task is registered on the corpora that annotate it.

.. list-table::
   :header-rows: 1
   :widths: 10 30 40 20

   * - Tag
     - Task
     - What a sample is classified as
     - Corpora
   * - ``asd``
     - Argumentative sentence detection
     - Argumentative or not. On UKDebates, a claim or not.
     - UKDebates, MM-USED
   * - ``acc``
     - Argumentative component classification
     - Claim or premise
     - MM-USED
   * - ``arc``
     - Argumentative relation classification
     - Support, attack, or neither, for a pair of sentences
     - M-Arg
   * - ``afc``
     - Argumentative fallacy classification
     - One of six fallacy categories: appeal to emotion, appeal to authority, ad hominem, false cause, slippery slope, and slogans
     - MM-USED-fallacy
   * - ``afd``
     - Argumentative fallacy detection
     - A dialogue sentence that a fallacious snippet holds, or not
     - MM-USED-fallacy

Input modes
-----------

A task reads one of three input modes, and the mode is a tag of its key.

1. ``text-only`` reads the transcript of each sentence.
2. ``audio-only`` reads the audio clip of each sentence.
3. ``text-audio`` reads both and fuses them inside the model.

A loader reads only what its input mode needs.
MM-USED and MM-USED-fallacy download the debate recordings and cut the clips only for a mode that reads audio.

Sentence pairs
--------------

Argumentative relation classification reads two sentences, and the label describes the relation between them.
A pair model applies one shared encoder to each side and concatenates the two outputs before its classification head.
The two sides are named ``a`` and ``b`` throughout, in the datasets, the batches, and the models.

Context
-------

MM-USED-fallacy can attach the previous sentences of a dialogue to each sample as context.
The ``with_context`` and ``context_window`` parameters of its loader control this.
The registered models do not read the context.
``VocabularyText`` counts the context texts when it fits its vocabulary, so the context can still add words to it.
