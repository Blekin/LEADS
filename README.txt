The `training_scripts` directory contains a dedicated training script for each data modality:

1. A standalone network for text;
2. A standalone network for grayscale images;
3. A standalone network for color images;
4. An audio-specific standalone network;
5. A Reformer-based model for text.

The QIA3 model for images is documented separately.

The `data` directory contains the raw files used in this study, including text, image, and audio files. Copyright in the text and audio materials remains with their original creators. These files are provided solely as illustrative examples and are not intended for commercial use. The authors captured all image files using personal mobile phones. Any identifiable individuals shown in the images consented to their use in this study.

`base_shifter.py` encodes and decodes DNA sequences using the base-shifter method. Both the input and output must consist exclusively of uppercase `A`, `C`, `G`, and `T` characters. No other characters are permitted.
