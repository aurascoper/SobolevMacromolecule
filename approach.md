Approach: Channel-Dropout Multiplex Cell-Phenotype Classification


The task and its two traps

Which of twelve mechanisms of action does this compound trigger? That is the task, and the competition hands you a three-panel Cell-Painting image to answer it, one of the three panels blacked out. You submit a probability for each class, and the score is log loss over a test set drawn from compounds that never appear in training. That last clause is where the difficulty lives. A model that learns to recognize the training compounds will look brilliant on the validation you run yourself and fall over the moment it meets a compound it has not seen.

Two traps sit inside this setup, and every decision below is built to step around one of them.


What the data actually is

There are 288 training images and 144 test images, each 480 by 160 pixels, laid out as three 160 by 160 panels side by side. Measuring the pixels instead of trusting the prose settled several questions at once. Every panel is a single fluorescence marker rendered through a fixed color lookup table: inside any live panel the red, green, and blue channels correlate above 0.998, so the color is decoration and the biology is one intensity map. Exactly one panel per image is pure black, zero in every channel, and the masked_region column names which one. That masking is identical in train and test, one panel gone, never zero, never two, and its position is close to uniform across the three slots.

The twelve classes are perfectly balanced by count, 24 images each. They are not balanced in the way that decides the outcome. Twenty-five compounds back those 288 images, and three classes, moa_02, moa_05, and moa_09, are each supported by a single compound. Hold onto that fact. It is the center of gravity for everything that follows.


Compound confounding

Picture the network looking at the 24 images that carry the moa_05 label. Every one of them is the same compound, stained on the same plates, under the same lamp. To the model, moa_05 and the batch signature of that one compound are not two things it can pull apart. There is no second compound in the class to average the nuisance away, so whatever the model decides moa_05 looks like, it is really deciding what that compound's staining artifact looks like. Then grading arrives carrying a compound the model has never seen, the artifact it memorized is gone, and the class it felt surest about becomes the class it is surest and wrong about.

Call these the thin classes. Their difficulty has nothing to do with how many images they hold and everything to do with holding only one compound, which leaves the model nothing to generalize from. You cannot manufacture compound invariance for moa_05 out of moa_05's own pictures, because there is exactly one compound in there to study. Whatever invariance those classes get has to arrive from somewhere else, either from the classes that do carry several compounds or from the backbone's prior. That single realization sets the shape of the whole system: the dense classes and the pretrained features supply the invariance, and the thin classes get protected rather than believed.


Why there is no channel-dropout augmentation to add

The name of the competition points straight at channel dropout, and the reflex is to teach the model to survive a missing panel by dropping panels at random during training. Here the reflex is wrong, and again the data is what shows it. Train and test already share the same masking, exactly one black panel on every image in both splits, so there is no distance between training masking and test masking for augmentation to close. Dropping a panel at random closes a gap that does not exist. And it does damage on the way: the masked panel is destroyed, zeroed at the pixel level, so you cannot slide the mask to a fresh position, and blacking out a second panel builds a two-missing image the test set never contains. The augmentation the title seems to ask for would train the model on a distribution it will never be graded against. Whatever regularization you want has to live inside the two surviving panels, gentle crops and flips and brightness, and it has to keep each panel in its slot so the mask label stays true.


The frozen backbone

With 288 images, fine-tuning a large vision model is mostly a faster way to memorize the training compounds. The features carry the load, so the backbone stays frozen and only a light head learns. Which backbone to use came down to a short experiment. A frozen convnext_tiny scored 1.463 log loss on the compound-blind out-of-fold set, convnext_small reached 1.366, and convnext_base slid back to 1.344. A bigger model losing to a smaller one is the tell of a dataset too thin to feed it. Convnext_small is the sweet spot, and the head reads its 768-dimensional features next to a three-way one-hot of which panel is dead.


Leakage-free cross-validation

Let a compound sit in both the training half and the held-out half of a split, and validation measures memorization while reporting it as generalization. So the folds are grouped by compound. No compound appears on both sides of a split, and out-of-fold accuracy turns into accuracy on compounds the model trained without.

The thin classes force a special case. Their one compound has to ride in every training fold, since a fold that held it out would hold zero examples of that class and could not learn it. The result is that the thin classes get no out-of-fold prediction at all, which looks like a hole and is really the honest answer: there is no leakage-free way to validate a class you can only ever train on. What the grouping hands you in return is a leave-one-compound-out test on the two-compound classes for free. Whenever one of their compounds lands in validation, the model trained only on the other, which is the nearest measurable stand-in for the thin-class condition.


Calibration

Under log loss a confident wrong answer costs far more than an unsure one, and the thin classes are precisely where confident wrong answers wait. Calibration is where they get their protection. A single temperature is fit on the dense-class out-of-fold logits, where the held-out-compound predictions are genuine. The thin classes cannot be calibrated on their own data, so their confidence is shrunk by a factor read off the two-compound classes and carried over as a lower bound, on the plain logic that a class with one compound needs at least as much damping as a class with two.

Notice what the sign of that shrink does. It is applied in probability space and renormalized, and that detail earns its place. An earlier version divided the thin-class logits by a temperature, but dividing a negative logit drags it toward zero, which raises that class's probability on exactly the images that are not thin-class. Shrinking the probability and renormalizing damps the class no matter the sign. Beyond that, the disagreement among the fold models, read as the mutual information between them rather than the entropy of their average, eases genuinely uncertain predictions toward the uniform prior.


Two bugs the design exists to prevent

Both are the sort that pass every local test and only announce themselves on the private leaderboard. Start with the class indices. An earlier plan protected classes 1, 4, and 8, having counted the second, fifth, and ninth class and quietly subtracted one, when the single-compound classes are moa_02, moa_05, and moa_09, which encode to 2, 5, and 9. That single off-by-one would have damped three healthy multi-compound classes and left the actual confounded ones fully exposed, with the code running green the entire time. The repair is to derive the thin classes from the data and assert the mapping, so the program halts the instant the data disagrees.

The second bug is the logit division from the calibration section. The two share a property worth saying out loud: each one yields a model that looks fine on your machine and fails right where the payout lives, on unseen compounds.


Results

The frozen convnext_small pipeline scores 1.366 log loss on the compound-blind out-of-fold set against 2.485 for the uniform prior, at a macro F1 of 0.495. Read torchvision for that figure; the same architecture through timm's weights reaches 1.316, the gap being nothing but pretraining and pooling. The full pipeline, feature extraction across 432 images, five folds, calibration, and inference, runs in roughly eighty seconds on a laptop CPU, and a fraction of that on the A10G, well inside the thirty-minute grading budget.


Eris submission mechanics

The graded artifact is one self-contained notebook. It reads its data from dataset/public, writes probabilities to working/submission.csv, calls only the libraries in the standard grading image, and finishes far under the thirty-minute limit on the A10G. Backbone weights load defensively, pretrained when the environment can reach them and a logged fallback when it cannot, so the notebook produces a valid file in either case. No language model runs anywhere in the pipeline; the classifier is convolutional from the first layer to the last.


What to watch next

The baseline already wears the failure mode it was built to fear. Across the 144 test predictions, moa_05, a thin class, is the top choice on 21 of them, more than its fair share, because the leave-one-compound-out estimate found no overconfidence to correct and therefore applied no damping. That estimate is a floor, not a final word, and a deliberate extra shrink on the thin classes is the cheapest hedge the next submission can buy. The platform pays for a visible climb from an honest baseline to a stronger one, so the order is fixed: ship the convnext_small model first, then spend the open compute budget. The safest thing you can do for a class you were never allowed to validate is refuse to be certain about it.
