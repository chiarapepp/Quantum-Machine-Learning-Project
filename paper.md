## Quantum Computing uantumEngineering

Received 6 November 2023; revised 24 January 2024; accepted 25 January 2024; date of publication 29 January 2024;
date of current version 1 March 2024.
_Digital Object Identifier 10.1109/TQE.2024._

# Network Anomaly Detection Using

# Quantum Neural Networks on Noisy

# Quantum Computers

ALON KUKLIANSKY^1 , MARKO ORESCANIN^1 (Member, IEEE),
CHAD BOLLMANN^1 (Senior Member, IEEE), AND THEODORE HUFFMIRE^1
Naval Postgraduate School, Monterey, CA 93943 USA
Corresponding author: Alon Kukliansky (e-mail: alon.kukliansky.is@nps.edu).
This work was supported by the Internal Funding of the Naval Postgraduate School (NPS) and in part by LP-CRADA under
Agreement LP-CRADA-NPS-23-0119 from IonQ, Inc. to NPS.

**ABSTRACT** The escalating threat and impact of network-based attacks necessitate innovative intrusion
detection systems. Machine learning has shown promise, with recent strides in quantum machine learning
offering new avenues. However, the potential of quantum computing is tempered by challenges in current
noisy intermediate-scale quantum era machines. In this article, we explore quantum neural networks (QNNs)
for intrusion detection, optimizing their performance within current quantum computing limitations. Our
approach includes efficient classical feature encoding, QNN classifier selection, and performance tuning
leveraging current quantum computational power. This study culminates in an optimized multilayered QNN
architecture for network intrusion detection. A small version of the proposed architecture was implemented
on IonQ’s Aria-1 quantum computer, achieving a notable 0.86 F1 score using the NF-UNSW-NB15 dataset.
In addition, we introduce a novel metric, certainty factor, laying the foundation for future integration of
uncertainty measures in quantum classification outputs. Moreover, this factor is used to predict the noise
susceptibility of our quantum binary classification system.

**INDEX TERMS** Intrusion detection, network intrusion detection system (NIDS), quantum neural network
(QNN).

**I. INTRODUCTION**
In recent years, the proliferation of network-based attacks
and security breaches has become a growing concern for
organizations and individuals alike. Traditional intrusion
detection systems (IDSs) often struggle to keep pace
with the ever-evolving threat landscape due to the increasing
complexity and sophistication of modern attacks[1],[2],[3].
Consequently, there is a pressing need for innovative
approaches that can enhance the accuracy and efficiency
of intrusion detection. Machine learning (ML) is one such
approach that has shown promise in enhancing the accuracy
and efficiency of IDSs[4],[5], [6], [7], [8]. ML-based
network intrusion detection systems (NIDSs) have proven to
be an essential, scalable tool in protecting networks against
cyberattacks.
With the emergence of quantum computational research,
the realm of quantum machine learning (QML) has gar-
nered attention. Noteworthy advancements in various QML
modalities, such as parameterized quantum circuits[9]and

```
variational learning[10], have paved the way for developing
efficient quantum classifiers[11],[12],[13].
While quantum computing presents immense potential,
it is not without its challenges, particularly in the context
of noisy intermediate-scale quantum (NISQ)[14]era ma-
chines. These devices are characterized by inherent imper-
fections stemming from a variety of sources, such as environ-
mental interference, control imprecision, and decoherence
effects[15].
Recent studies[16],[17],[18],[19]have substantiated
the efficacy of quantum neural network (QNN) and quan-
tum support vector machines (QSVMs) in classifying net-
work activities. These findings have demonstrated exception-
ally high detection rates in noiseless simulation, showcas-
ing the promise of quantum approaches in the field of net-
work security and intrusion detection; however, none of them
have shown good classification performance on a physical
quantum computer.
```
© 2024 The Authors. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License.
VOLUME 5, 2024 For more information, see https://creativecommons.org/licenses/by-nc-nd/4.0/ 3100611


## uantumEngineering

```
Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS
```
This work also investigates classifying network
activities with a QNN. We began with a survey of
various established QNN classifiers, culminating in the
selection of the classifier demonstrating the most promising
classification performance. But rather than attempting to
improve performance by mitigating noise, we pursued
the creation of an ultralean circuit, designed to suffer the
least possible interference. This strategy seeks to harness
the power of quantum computing while working within
its current limitations. This deliberate choice is central to
our exploration, as it allows us to push the boundaries of
quantum-enhanced intrusion detection, even within the
constraints of today’s quantum computing limitations.
Another contribution of our approach lies in the develop-
ment of an efficient procedure to encode classical NetFlow
information from the NF-UNSW-NB15 dataset [20].This
procedure minimizes the quantum resources required, ulti-
mately leading to a one-qubit representation for each classi-
cal feature and a single rotation gate for its state preparation.
We conclude by successfully testing our novel algorithms
and achieving a significant improvement over the state of the
art using IonQ’s quantum computers.
The key contributions made in this research include the
following.

```
1) Development of an ultralean QNN architecture, opti-
mized for NISQ devices, achieving network activity
classification performance on par with classical ML
techniques
2) A novel encoding scheme to efficiently represent clas-
sical NetFlow data in qubit rotations
3) The introduction of a certainty factor that enriches the
analysis by evaluating noise’s impact on classification
performance and allowing for nuanced assessments be-
yond binary classification
4) Advanced the state-of-the-art quantum classification
F1 score on a quantum computer from 0.838 to 0.
with a simplified architecture (6 layers versus 8).
```
The rest of this article is organized as follows.
Section II introduces QNN, our chosen dataset, and
related work. SectionIIIprovides a detailed description
of our research methodology. The results are presented in
SectionIV, and the discussion of the results is presented
in SectionV. Finally, SectionVIsummarizes a concise
overview of the main points discussed in this article.

**II. BACKGROUND**
_A. QUANTUM NEURAL NETWORKS_
QNNs manipulate qubits that hold quantum information.
Qubits can exist in the standard basis states| 0 〉or| 1 〉,or
in a superposition of both, denoted as a general state|ψ〉.
This superposition enables qubits to encode more complex
quantum information compared with classical binary bits.
A convenient representation of qubit states is the Bloch
sphere (see Fig. 1 ), where points on the sphere correspond to

```
FIGURE 1. Bloch sphere—A qubit state is represeted as
|ψ〉=cos(θ 2 )| 0 〉+ ei φsin(θ 2 )| 1 〉
```
```
state vectors|ψ〉=cos(θ/2)| 0 〉+ ei φsin(θ/2)| 1 〉. Here, the
angular coordinatesθandφparameterize the superposition
of| 0 〉and| 1 〉that defines the qubit state|ψ〉. For example,
θ=0 corresponds to the| 0 〉state whileθ=πindicates| 1 〉.
Fig. 1 provides a graphical depiction of representing qubits
on the Bloch sphere.
Before employing a QNN on classical information, the ini-
tial quantum state must be prepared according to the classical
features, also known as state preparation. This initialization
of the input qubits to a desired starting state is a key compo-
nent of nearly all quantum algorithms[21].
The training process for QNNs has conceptual parallels
to classical neural networks in utilizing feedforward and
backpropagation techniques. However, the nature of what
is learned during training differs fundamentally. For classi-
cal networks, training primarily involves adjusting weight
and bias parameters to optimize performance. In contrast,
quantum network training shifts towards fine-tuning quan-
tum gates parameters that represent rotations and the type
of entanglement operations performed on the qubits. Fig. 5
shows several known QNN architectures. A recent review
paper on quantum classifiers and QNN can be found in[22].
```
```
B. DATASET
The UNSW-NB15 dataset[8],[23],[24],[25],[26]is a
widely utilized[27],[28],[29],[30]compilation of gener-
ated network traffic data, specifically designed for evaluat-
ing NIDSs. It provides a realistic and diverse representation
of network behavior, containing normal traffic as well as
various types of attacks. Sarhan et al.[20]transformed the
packet-based UNSW-NB15 dataset into a NetFlow dataset,
denoting it as NF-UNSW-NB15. The conversion was un-
dertaken to establish a standardized feature set, facilitating
the consistent comparison of ML architectures across diverse
NIDS datasets, and ultimately merging all different datasets
into a single NetFlow-based dataset.
The NF-UNSW-NB15 dataset has been used in several
ML studies[31],[32],[33], and provides a robust foundation
for our QML investigation. It has 1 623 118 labeled NetFlow
samples, where 72 406 (4.4%) are malicious and each sample
has 12 features. To avoid biasing and to further reduce the
number of features we dropped the source and destination IP
```

## Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS uantumEngineering

**TABLE 1** Selected Features From the NF-UNSW-NB15 Dataset Including
the Range of Each Selected Feature and the Number of Its Unique Values

**FIGURE 2.** Log scale histograms of the selected features extracted from
the NF-UNSW-NB15 dataset. (a) IPv4 protocol. (b) Layer 7 protocol. (c)
TCP flags. (d) Out packets count. (e) In packets count. (f) In bytes count.
(g) Out bytes count. (h) Flow duration.

and port features, leading us to consider only eight features.
An overview of selected features and dataset statistics are de-
tailed in Table 1 and in Fig. 2. The dataset was complete and
had no missing values. The layer 7 protocol feature was clas-
sified by us as float since it has the structure of “<Number
>.<Number>.” Although the TCP flags consist of distinct
eight flags, we have only observed 15 unique values for that
feature.
In addition, given the natural imbalance nature of network-
ing data, we resampled the benign NetFlows to produce a
balanced dataset. For that, we used the resample utility of
scikit-learn with _random_state_ =123. The final dataset used
for our model development is balanced with 144 812 samples
and eight features. Also using scikit-learn, we implemented
a 15%–85% test-train split with _random_state_ =1.

_C. RELATED WORK_
Several studies considered the use of QML for the purpose
of anomaly detection in computer networks. For instance,
the Moore dataset[34]was used in[16]to train a single
layer QNN, consisting of a single two-qubit general unitary
acting on every feature qubit and the result qubit. Similar to

```
our approach, they produced a balanced dataset and normal-
ized the feature values to be in the [0,1] range, followed by
a single-qubit binary encoding. Their experimentation used
16 features in the first trial and 12 features in the second,
yielding F1 scores of 0.8 and 0.77, respectively.
A different study[17]used a quantum convolutional neu-
ral network (QCNN) and a QSVM to perform multiclass
classification. They synthesized their dataset by creating
streams out of packets generated by nmap and hping. Each
stream had 58 features, and all were encoded into a single
qubit by successive rotation around the x -axis. They reported
an accuracy 0.98 in simulation for the QCNN and also for the
QSVM.
The NSL-KDD[35]and the UNSW-NB15 datasets were
used in[18]to train a classical support vector machine
(SVM) and a QSVM. The reported QSVM accuracies in sim-
ulation for the NSL-KDD and UNSW-NB15 datasets were
0.92 and 0.64, while the classical SVM achieved accuracies
of 0.93 and 0.75, respectively. Notably, the training process
utilized only 150 data samples.
The only reported quantum-based intrusion detection,
with model evaluation tested on a quantum computer, is[19].
The authors subsampled the KDD CUP99 dataset[36]to
obtain 700 balanced training and 300 balanced test samples.
Previous studies on feature importance for this dataset led
them to select five out of the original 41 features. The fea-
tures were encoded using a Hadamard gate followed by a
Z rotation. The proposed QNN architecture had eight layers
of arbitrary single-qubit rotations and a staggered controlled
not (cnot) ring. In a noiseless simulation, they achieved a
0.983 F1 score; running on IBM’s noisy quantum computers
and averaging over 1000 shots for a test set containing 100
samples, they achieved an F1 score of 0.838.
In the original NF-UNSW-NB15 paper Sarhan et al.[20]
employed the extra trees classifier to perform binary classi-
fication and reported 0.986 accuracy and 0.85 F1 score. A
subsequent study[33]surpassed these results with F1 scores
of 0.9 and 0.92 using novelty and outlier detection. Another
work[31]reported an impressive accuracy of 0.998, though
it did not provide an F1 score. The best F1 score is docu-
mented in[32], where they combined NF-UNSW-NB15 with
other NetFlow NIDS datasets into a single dataset named
NF-UQ-NIDS. They used the unified dataset to train differ-
ent classifiers and calculated F1 scores for each classifier
and original dataset. For the NF-UNSW-NB15 dataset, they
achieved F1 scores between 0.87 and 0.987 depending on the
classifier. The best performance was achieved using a deep
neural network (DNN) with five dense hidden layers, nine
feature input layers, and a 21-neuron output layer.
Feature analysis of the NF-USNW-NB15 dataset has been
conducted in[37]. Three distinct algorithms were employed
to ascertain the importance ranking of each feature, which
were subsequently evaluated using both DNN and random
forest classifiers. The findings revealed that all eight fea-
tures were requisite for optimal accuracy performance with
the DNN, whereas the random forest classifier demonstrated
```

## uantumEngineering

```
Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS
```
**FIGURE 3.** Flowchart illustrating the training and testing process of various QNNs. The workflow includes feature filtering and benign NetFlow
resampling, dataset splitting, creation of classical-to-quantum encoding tables, training of networks using the TensorFlow-Quantum package, and
performance evaluation in both a noisy simulation and on IonQ’s quantum computers.

peak accuracy performance with just four features–namely,
TCP flags, in byte count, and out byte count. In addition, the
study highlighted instances where utilizing all features may
not be advantageous.
It is important to note that the NF-UNSW-NB15 dataset is
imbalanced. Not all studies have addressed this imbalance
through resampling, leading us to focus on the F1 score,
which provides an equal-weighted combination of precision
and recall. Due to the lack of standardization of splits (e.g.,
which samples constitute training versus testing) results vary
over reported studies. Therefore, reproducing and comparing
results is a nontrivial task. In SectionII-Bwe provided details
on how we performed the resampling and splitting of the
dataset, aiming to enhance comparability with future studies.

**III. METHODOLOGY**
This section explains in detail our research methodology.
In SectionIII-Awe introduce an approach for encoding
classical features into qubits. We then provide an overview
of different QNN architectures we evaluate in this work in
SectionIII-B. To enhance the analysis of the performance
we suggest a new performance metric, certainty factor, and
provide details on its definition and the intuition behind it
in SectionIII-C. Details of the training process including
hyperparameters are provided in SectionIII-EandIII-D.In
SectionIII-Fwe detail the methodology for evaluation of the
best performing QNN architecture through a realistic simu-
lation with a noise model as well as evaluation on the IonQ’s
quantum computers. A high-level overview of the research
process can be seen in Fig. 3.

```
FIGURE 4. State preparation circuit used for the qubit encoding of the
classical features. Each feature value is projected to an angle in a
quantized range [0,π], ensuring a minimum granularity of 0. 25 ◦.
```
```
A. DATA ENCODING
In order to use classical features in a QNN, the features need
to be encoded into quantum information stored in qubits.
Comparing several ways to perform the encoding[5],[13],
we decided to encode each feature value in a different qubit,
similarly to qubit encoding in[13]. In our IDS, we have a
limited number of features, hence we chose a single qubit
per feature. This approach avoids the need for a more in-
tricate entangled state that would use fewer qubits but re-
quire complex state preparation, which introduces undesired
noise.
For a given dataset S ={( x i , yi )} Si = 1 , we encode each
feature xi [ j ] (where j represents the feature and i de-
notes the sample) as a rotation around the x -axis. To
perform the rotation we first project each feature onto
an angle θ i [ j ]∈[0,π]. Then, we executed an Rx (θ i [ j ])
on each feature qubit, transforming it from the de-
fault| 0 〉state intoψ i [ j ]=cos(θ i [ 2 j ])| 0 〉+sin(θ i [ 2 j ])| 1 〉;see
Fig. 4.
```

## Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS uantumEngineering

A naive way to perform the projection is to normalize the
feature space, with the formula

```
θ i [ j ]=π
```
```
x i [ j ]
max k x k [ j ]
```
### .

Although attractive due to simplicity, this approach may
create extremely fine rotations. In practice, such an encod-
ing schema is impossible to generate with current quantum
hardware due to imperfections in the implementation of the
quantum gates that limit the rotation accuracy.
In order to mitigate this issue, we avoid fine rotations by
projecting each feature space to a quantized [0,π] range,
where we limit the rotation’s granularity to 0. 25 ◦. For fea-
tures that have a limited number of unique values (see
Table 1 ), we split the [0,π] range according to the number
of unique values, and then assign a projected value categori-
cally. For features that have many unique values, the projec-
tion from the feature space to [0,π] was done by binning the
values according to their percentile. For example, if we were
to split the range [0,π] into 100 bins, then a feature value
that is in the _x_ percentile will be projected to _x_ · 100 π.

_B. QNN ARCHITECTURES_
In this work, we investigated four different QNN architec-
tures: TTN[13], MERA[13], Quantum Convolutional Neu-
ral Network (QCNN)[12], and a simple architecture built
by stacking two-qubit Pauli rotation gates blocks, similar
to what was used in[38]. We refer to that architecture as
“Simple”; we illustrate all architectures in Fig. 5. _Ui_ and _Di_
represent a general quantum gate. In the QCNN architecture,
see Fig.5(d), all the gates with the same index have identical
parameters.
In each of the architectures, we modeled a generic two-
qubit gate using two generic one-qubit gates followed by
a cnot, as shown in Fig. 6. We note that one needs three
cnots to accurately decompose any general two-qubit gate
into cnots and general one-qubit gates[39]. However, our
goal is to design a QNN that can run on NISQ computers,
hence we chose to have only a single cnot as a balance
between the expressibility of the gate and its fidelity on a
noisy quantum computer.
All evaluated architectures have a single qubit output,
which we measure over many shots, and the majority vote of
these measurements is the network’s prediction. In a noise-
less simulation, we can look at the state of the output qubit
and predict accordingly. The result qubit in the “Simple”
architecture is initialized in the|−〉state and measured at the
end in the _x_ -basis, while in all the other architectures _z_ -basis
measurement is used.

_C. CERTAINTY FACTOR_
We introduce a confidence metric, referred to as the certainty
factor and denotedC, to quantify the model’s inherent sharp-
ness and degree of separation between the predicted class
probability distribution versus other classes. This continuous
measure of predictive confidence ranges from−1to1for

```
FIGURE 5. Evaluated QNN architectures. Each generic two-qubit gate
was modeled using two generic one-qubit gates followed by a Controlled
Not (CNOT),asshowninFig. 6. Each of the eight features was encoded as
a rotation around the x -axis for qubits 0–7. In (c) the Res qubit is
initialized to the|−〉state, and measured in the x -basis, while all the
other measurements are z -basis. (a) TTN. (b) MERA. (c) “Simple.”
(d) QCNN.
```
```
FIGURE 6. Two-qubit gate modeled by two generic one-qubit gates and
aCNOT.
```
```
each sample, enabling analysis of prediction quality on a
per-sample basis.
Specifically, the certainty factor approximates the rela-
tive entropy or divergence between the predicted outcome
probability distribution and an ideal fully confident singular
distribution. For a labeled True example where the model
```

## uantumEngineering

```
Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS
```
**LISTING 1:** Listing 1: Custom accuracy function that correctly handles
the±1 labels[38].

predicts a|ψ〉=α 0 | 0 〉+α 1 | 1 〉state, with| 0 〉expected for
true, the certainty factor is defined asC=|α 0 |^2 −|α 1 |^2.
When C=1, the predicted distribution is maximally
peaked on the expected outcome, equivalent to a fully con-
fident singular distribution. IfC=−1, the distribution is
inverted and predictions will be incorrect. ForC=0the
distribution is maximally flat, indicating completely random
guesses. As|C|increases from 0, the predicted distribution
sharpens, reflecting greater confidence.
By tracking certainty factor distributions, we can thus
assess the degradation of predictive sharpness and separation
margins as noise intensifies. C identifies vulnerable
predictions likely to be impacted by errors before noise
is introduced.

_D. LEARNING INFRASTRUCTURE_
In this work, we used TensorFlow-Quantum[40]v0.7.2 with
Python 3.8, and TensorFlow v2.7.0. Using KERAS, we per-
formed the learning procedure and used the Hinge loss func-
tion[41], not before converting the True and False labels
to 1 and−1. For the accuracy metric, we used a custom
accuracy function that correctly handles a set of±1 labels,
see Listing 1.

_E. HYPERPARAMETERS SEARCH_
In order to gain insight into the performance and trainability
of the different architectures, we performed a detailed
grid search over hyperparameters. We tried batch sizes
of 16 and 32. The different learning rates were 0. 1 , 0. 05 ,
0. 02 , 0. 015 , 0. 01 , 0. 005 ,and 0.001. We used two different
optimizers, Adam [42]and stochastic gradient descent
(SGD), with momentum 0, 0. 2 , 0. 3 , and decay rate
0 , 0. 001 , 0 .01. Finally, we tried all four architectures,
and in the “Simple” architecture we tried 1, 2, 4, and 6
layers. For each layer, we tried ZZ XX YY; ZZ XX; XX YY;
and ZZ YY as the possible entanglement rotations.

_F. EVALUATING THE BEST ARCHITECTURE WITH NOISE_
After we identified that the “Simple” QNN architecture
had the best performance in the noiseless simulation (see
SectionIV-A), we decided to see whether and how noise

```
affects the performance of our classification system. We
choose to use IonQ’s harmony quantum processing unit
(QPU)[43]as our target hardware because its native gates
match perfectly with the Rxx and Ryy gatesusedinthe
“Simple” network. Moreover, the ion trap has all-to-all
connectivity that is heavily used in our QNN architecture.
```
```
1) HARMONY EVALUATION
As a first step, we used IonQ’s noisy simulator to retest the
QNN performance. This required creating a different quan-
tum circuit for every sample in our test set. These circuits
are a concatenation of two circuits—the feature embedding
circuit (state preparation) and the network itself. We use the
serialization capabilities of Cirq-IonQ Python package to^1
convert the combined circuit to JSON format. Next, we use
IonQ’s REST API to send the inference simulation tasks and,
finally, retrieve the results using the same API.
The last evaluation step was to run the inference circuits
on the harmony machine. It is very expensive to run many
circuits with many shots on a physical quantum computer;
thus, we decided to evaluate our proposed QNN using a
smaller random sample of our test set. Due to our funding
sources, the experiments on the QPU were done via Amazon
Braket Cloud API and not directly using IonQ’s RESET API.
In our first trial, we ran a balanced test set of 60 samples with
120 shots per task. The second trial consisted of 40 samples
and 200 shots per task.
```
```
2) ARIA-1 EVALUATION
The harmony experiments yielded disappointing results,
demonstrating low classification performance as discussed
in SectionIV-C. After contacting IonQ’s experts, we were
advised to try their error mitigation techniques[44],[45]or
to use the newer QPU Aria-1[46]for its improved noise error
rates. We decided to experiment with the newer hardware.
Aria-1 is also an all-to-all machine that has the same native
gates as harmony, so it was a natural fit for our “Simple”
QNN architecture.
Similar to the harmony experiments, our first step was to
evaluate the QNN on IonQ’s noisy simulator. To that end, we
used the same procedure described in SectionIII-F1.Next,
we again used Amazon Braket to run on the QPU itself. To
our surprise, we were not able to run our six-layer architec-
ture on Aria-1.
With support from IonQ, we determined that our circuits
exceeded the current limit for the number of unique gates al-
lowed in a single channel. We considered consolidating gates
with close rotations or evaluating a shallower architecture.
We chose the latter, evaluating a two-layer XY architecture
with 200 shots per task. Keeping our limited resources in
```
(^1) https://github.com/quantumlib/Cirq/tree/master/cirq-ionq


## Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS uantumEngineering

**TABLE 2** Comparison of the Best F1 Score and Average Convergence
Epochs for Different Network Architectures and Optimizers in a
Noiseless Simulation

**TABLE 3** Simple Architecture F1 Score Results for Different Number of
Layers and Their Type in a Noiseless Simulation

**TABLE 4** Best F1 Score Comparison Between Adam and SGD Optimizers
for Different Learning Rates in the Simple Architecture in a Noiseless
Simulation

mind, we began with a fairly small balanced test set, ul-
timately increasing to 776 balanced samples (3.5% of our
complete test set).

**IV. RESULTS**
In this section, we present and discuss the results of our
research in the following sections. SectionIV-Aholds the
key results from our noiseless architecture exploration exper-
iments. In SectionIV-B, we evaluate our simple architecture
on IonQ’s harmony and Aria-1 noisy simulators. We analyze
noise impact on classification using the introduced certainty
factor. Then, in SectionsIV-CandIV-Dwe set forth the
results from our Harmony and Aria-1 QPU experiments.

_A. NOISELESS QNN ARCHITECTURE COMPARISON_
Tables 2 – 4 summarize our findings from architecture explo-
ration experiments. Table 2 gives the best F1 score value
obtained for each QNN architecture in our hyperparameter
search. It demonstrates that the simple architecture has out-
performed all the other architectures with an F1 score of
0.907. In addition, as given in Table 3 , the F1 score improves
with an increase in network depth. Moreover, one can also
observe that the XY layer type achieves the best performance
compared to the other layer types.
To assess the convergence speed of different architectures,
we determined the last epoch, at which the loss exhibited

```
FIGURE 7. Violin plots depicting the distribution of certainty factor for
our six-layer Simple QNN architecture in three simulation modes:
noiseless, Aria-1 noise model, and harmony noise model. The green
dashed line represents zero certainty, where everything above it is a
correct prediction while everything below is a wrong prediction. It is
clear to see from the figure that as we have more noise, the distribution
is squashed into zero certainty, and as expected Harmony has more
noise compared to Aria-1.
```
```
improvement. The results are summarized in Table 2 , demon-
strating that the Adam optimizer achieves significantly faster
convergence compared with the SGD optimizer. Further-
more, both optimizers outperform classical neural networks
in terms of convergence speed. A comparison between dif-
ferent learning rates coupled with the optimizer used for
the training of the simple architecture is given in Table 4.
Overall, Adam converges faster, but SGD yields a better
classification network.
Ultimately, the best F1 score was achieved using the
following hyperparameters.
```
```
1) Optimizer—SGD with a decay of 0.001.
2) Learning rate—0.02.
3) QNN architecture—Simple with six layers of type XY.
4) Batch size—32.
```
```
We did not observe any obvious benefit from using a batch
size of 16 versus 32.
```
```
B. NOISY SIMULATION AND CERTAINTY FACTOR
The F1 scores obtained on IonQ’s noisy simulator for our
six-layer architecture using harmony and Aria-1 noise mod-
els are 0.8789 and 0.886, respectively, whereas the noiseless
simulation had an F1 score of 0.907.
To analyze the noise impact on the performance of our
QNN predictions, we use the certainty factor, which we de-
fined in SectionIII-C, and plot its distribution violin plot for
the noisy and the noiseless simulation on our test set. Fig. 7
shows the comparison; from it, we can see that as the noise
increases the distribution is squeezed toward the middle. This
symmetric deformation is expected, as the noise should have
the same effect on correct and wrong predictions.
```

## uantumEngineering

```
Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS
```
**FIGURE 8.** Histogram of the 776 Aria-1 inference runs, binned according
to their original certainty factor in the noiseless simulation. The green
and red bins represent correct and incorrect Aria-1 inference predictions,
respectively. The figure demonstrates that samples with a certainty factor
close to zero are more likely to be affected by noise and result in a
classification flip.

**TABLE 5** Confusion Matrices for Both of the Harmony Experiments

Fig. 8 shows the impact of noise on the classification in our
776 Aria-1 inference runs using the two-layer simple QNN
architecture. It demonstrates how samples with a certainty
factor closer to zero are more likely to be affected and flip
the noiseless prediction. The certainty factor can predict the
susceptibility of a prediction. In the absence of noise, and
with a sufficient number of shots, all the bins corresponding
to a positive certainty factor would be green, while those
corresponding to a negative certainty factor would be red.

_C. HARMONY EXPERIMENTS_
In our first experiment, we used a balanced test set of 60
examples. For each of them, we ran a 120-shot task. Af-
ter seeing the disappointing results, we decided to run an-
other 40 examples and increased the number of shots to 200.
Table 5 gives the confusion matrices for both of our harmony
experiments.
Following another set of disappointing results, we wanted
to rule out the possibility that the source of the discrepancy
between the simulation and the quantum computer has to
do with rotation angle precision. Fig. 9 shows the noiseless
simulated prediction performance as we decrease the number
of digits after the decimal point. The graph clearly shows a
degradation in F1 performance, but not so severe as we see
while running on harmony QPU. Thus, we conclude that the
introduced error is the result of overall quantum computer

```
FIGURE 9. Noiseless Harmony classification performance as a function
of the number of decimal points used in the quantum gate rotation
angles, plotted for different numbers and types of layers. The
performance degradation is most significant when transitioning from
two decimal places to one.
```
```
TABLE 6 F1 Score Comparison of Our two-Layer Architecture for the
Complete Test Set and the Randomly Selected Subset of 776 Examples
```
```
TABLE 7 Confusion Matrices for the Aria-1 Noisy Simulation and
Hardware Experiments
```
```
noise error vice a loss of precision when moving from simu-
lation to instantiation. As discussed above, these poor results
and discussions with IonQ led us to port our experiment to
Aria-1.
```
```
D. ARIA-1 EXPERIMENTS
Table 6 presents a summary of the F1 score performance
achieved by our two-layer architecture. The evaluation en-
compasses both the complete test set and a randomly selected
subset of 776 examples. The scenarios include a noiseless
simulation, a noisy simulation utilizing IonQ’s Aria-1 noise
model, and execution on the Aria-1 machine. Notably, it is
worth mentioning that the F1 scores in the noiseless simula-
tion for both the complete test set and the random subset are
nearly identical. Furthermore, the performance in the noisy
simulation serves as a good approximation to the results ob-
tained on the Aria-1 machine. Table 7 presents the confusion
matrices for the subset classification on the noisy simulation
and the Aria-1 machine.
```

## Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS uantumEngineering

**TABLE 8** F1 Score Comparison Between This Work and Other Classical and Quantum NIDSs Found in the Literature

As discussed in SectionIII-F, we were only able to simu-
late our proposed six-layer simple architecture; this six-layer
architecture yielded an F1 score of 0.886 using Aria-1’s noisy
simulator. Given the strong alignment between the two-layer
simulation and the QPU experiment results, we anticipate
being able to reproduce this classification performance in
the six-layer architecture on Aria-1 once IonQ implements
the necessary enhancements to run circuits with more unique
two-qubit gates per channel.

**V. DISCUSSION**
The results of our study demonstrate a significant advance-
ment in the accuracy of quantum computing-based classifi-
cation for network intrusion detection. Through the imple-
mentation of a simple QNN classifier, we achieved F1 score
performance comparable to those obtained by a more com-
plex classical ML techniques[20],[33],[32]; see Table 8.
It is noteworthy to mention that some of the existing works
in this domain utilized different datasets, and among those
that employed the same dataset as ours, the specific splits
for training were not specified. Moreover, some studies[32]
have tested against the NF-UNSW-NB15 dataset but trained
using a combination of datasets. Our achievement not only
underscores the potential of QML but also demonstrates the
practicality of employing quantum algorithms in real-world
cybersecurity applications.
Furthermore, our success in instantiating the classifier on a
current NISQ-era machine was enabled by a combination of
innovative classical feature encoding techniques and the lean
architecture of our custom QNN. The strategic choice of an
all-to-all quantum computer, whose native gates were a per-
fect fit with our QNN’s _Rxx_ and _Ryy_ quantum gates, ensured
minimal performance degradation attributable to noise.
Incorporating the certainty factor into the analysis, pro-
vided a deeper understanding of how classification perfor-
mance is affected by noise. This novel metric enabled us
to systematically analyze the impact of quantum noise on
classifier performance at the system level.
Moreover, the introduction of the certainty factor opens
up intriguing possibilities for future research endeavors. This

```
factor can serve as a foundation for extending the classifica-
tion output to incorporate measures of uncertainty, akin to
the principles underlying Bayesian networks. By integrating
uncertainty assessments, we move beyond binary classifi-
cations, allowing for nuanced interpretations of the results.
The certainty factor not only enriches the depth of infor-
mation provided by our intrusion detection system but also
lays the groundwork for more sophisticated decision-making
processes in complex and dynamic network environments.
Such an advancement holds promise for advancing the state
of the art in intrusion detection, which would lead to more
resilient and adaptive cybersecurity architectures.
Throughout our experimentation, we worked within an
interesting constraint. Given our limited computational re-
sources, we found ourselves tasked with the decision of allo-
cating an optimal number of shots for each inference process.
This scenario encapsulates an interesting tradeoff: allocating
more shots per inference to enhance confidence levels for in-
dividual results, or distributing shots across more inferences,
thereby maximizing the overall reliability of our evaluation
procedure. To address this challenge, we adopted a random-
ized approach to sample our testset using various seeds and
sample sizes. Using the statistical insights derived from our
noisy simulation enabled us to make informed decisions
about the allocation of shots. In addition, in an operational
setting, one might monitor the certainty factor and adaptively
run additional shots in case of low confidence predictions.
The current methodology could be improved in several
ways to further optimize performance. One such improve-
ment is using fewer features from the dataset, balancing the
information gained from every feature addition with the in-
troduced noise from a more complex QNN. One might use
the feature importance analysis done in[37].
Another avenue could be to utilize multiple QNNs where
through bagging and random feature selection, similar to
random forest algorithm, an ensemble of weak classifiers
could outperform a strong classifier (e.g., larger QNNs) op-
erating on all features. By effectively leveraging the parallel
computational power of quantum machines in this ensemble
framework, we could further amplify the efficacy of our
```

## uantumEngineering

```
Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS
```
intrusion detection system. Finally, as suggested by IonQ,
implementing noise mitigation strategies represents another
possible avenue for classification performance improvement.

**VI. CONCLUSION**
This article represents a significant advancement in the
utilization of quantum computing for intrusion detection.
Compared with sophisticated classical ML models we have
achieved similar F1 score performance while having fast con-
vergence, indicating the practical potential of quantum algo-
rithms in real-world security applications. We are also among
the pioneers in leveraging QNN on a physical quantum com-
puter for network intrusion detection, potentially achieving
the highest performance known to date. Through innovative
feature encoding techniques and a streamlined QNN archi-
tecture tailored to a compatible quantum platform, we have
mitigated deterioration in performance caused by noise. The
introduction of a certainty factor enriched our analysis, of-
fering insights into the impact of noise, and paves the way
for incorporating uncertainty measures into future research,
yielding advances in intrusion detection and cybersecurity
frameworks and in QML in general. Moving forward, av-
enues for future work include exploring alternative features
from NetFlow data, employing multiple QNNs in an ensem-
ble fashion, and implementing noise mitigation techniques.

**REFERENCES**
[1] “2023 Global Threat Report|CrowdStrike,” [Online]. Available:
https://www.crowdstrike.com/global-threat-report/
[2] “2023 Data Breach Investigations Report,” [Online]. Available:
https://www.verizon.com/business/resources/reports/dbir/
[3] K. Shaukat, S. Luo, V. Varadharajan, I. A. Hameed, and M. Xu, “A survey
on machine learning techniques for cyber security in the last decade,”
_IEEE Access_ , vol. 8, pp. 222310–222354, 2020, doi:10.1109/AC-
CESS.2020.3041951.
[4] Y. Liu, X. Wang, S. Li, and J. Wu, “Machine learning and deep learning
methods for intrusion detection: A review,” _Appl. Sci._ , vol. 9, no. 20, 2019,
Art. no. 4396, doi:10.3390/app9204396.
[5] S. Kaddoura, “Classification of malicious and benign websites
by network features using supervised machine learning algo-
rithms,” in _Proc. 5th Cyber Secur. Netw. Conf._ , 2021, pp. 36–40,
doi:10.1109/CSNet52717.2021.9614273.
[6] P. Podder, S. Bharati, M. R. H. Mondal, P. K. Paul, and U. Kose, “Arti-
ficial neural network for cybersecurity: A comprehensive review,” 2021,
doi:10.48550/arXiv.2107.01185.
[7] M. Rabbani et al., “A review on machine learning approaches for network
malicious behavior detection in emerging technologies,” _Entropy_ , vol. 23,
no. 5, May 2021, Art. no. 529, doi:10.3390/e23050529.
[8] V. Kanimozhi and P. Jacob, “UNSW-NB15 dataset feature selection and
network intrusion detection using deep learning,” _Int. J. Recent Tech-
nol. Eng._ , vol. 7, no. 5S2, pp. 443–446, Jan. 2019. [Online]. Available:
https://www.ijrte.org/portfolio-item/ES2080017519/
[9] M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, “Parameterized quan-
tum circuits as machine learning models,” _Quantum Sci. Technol._ ,vol.4,
no. 4, Nov. 2019, Art. no. 043001, doi:10.1088/2058-9565/ab4eb5.
[10] F. Tacchino et al., “Variational learning for quantum artificial neu-
ral networks,” _IEEE Trans. Quantum Eng._ , vol. 2, pp. 1–10, 2021,
doi:10.1109/TQE.2021.3062494.
[11] W. Li, Z. Lu, and D.-L. Deng, “Quantum neural network classifiers:
A tutorial,” _SciPost Phys. Lecture Notes_ , Aug. 2022, Art. no. 61,
doi:10.21468/SciPostPhysLectNotes.61.

```
[12] I. Cong, S. Choi, and M. D. Lukin, “Quantum convolutional neural
networks,” Nature Phys. , vol. 15, no. 12, pp. 1273–1278, Dec. 2019,
doi:10.1038/s41567-019-0648-8.
[13] E. Grant et al., “Hierarchical quantum classifiers,” NPJ
Quantum Information , vol. 4, no. 1, Dec. 2018, Art. no. 65,
doi:10.1038/s41534-018-0116-9.
[14] J. Preskill, “Quantum computing in the NISQ era and beyond,” Quantum ,
vol. 2, Aug. 2018, Art. no. 79, doi:10.22331/q-2018-08-06-79.
[15] N. P. de Leon et al., “Materials challenges and opportunities for quan-
tum computing hardware,” Science , vol. 372, no. 6539, Apr. 2021,
Art. no. eabb2823, doi:10.1126/science.abb2823.
[16] M. Zhang, B. Lv, and Z.-S. Liu, “Network attack traffic recognition based
on quantum neural network,” in Proc. 7th Int. Conf. Comput. Intell. Appl. ,
2022, pp. 71–75, doi:10.1109/ICCIA55271.2022.9828461.
[17] M. Kalinin and V. Krundyshev, “Security intrusion detection using quan-
tum machine learning techniques,” J. Comput. Virol. Hacking Techn. ,
vol. 19, pp. 125–136, Jun. 2022, doi:10.1007/s11416-022-00435-0.
[18] A. Gouveia and M. Correia, “Towards quantum-enhanced machine learn-
ing for network intrusion detection,” in Proc. IEEE 19th Int. Symp. Netw.
Comput. Appl. , 2020, pp. 1–8, doi:10.1109/NCA51143.2020.9306691.
[19] C. Gong, W. Guan, A. Gani, and H. Qi, “Network attack detection scheme
based on variational quantum neural network,” J. Supercomputing , vol. 78,
no. 15, pp. 16876–16897, Oct. 2022, doi:10.1007/s11227-022-04542-z.
[20] M. Sarhan, S. N. Layeghy Moustafa, and M. Portmann, “NetFlow
datasets for machine learning-based network intrusion detection sys-
tems,” in Big Data Technologies and Applications, ser. Lecture Notes
of the Institute for Computer Sciences, Social Informatics and Telecom-
munications Engineering , Z. Deze, H. Huang, R. Hou, S. Rho, and
N. Chilamkurti, Eds. Berlin, Germany: Springer, 2021, pp. 117–135,
doi:10.1007/978-3-030-72802-1_9.
[21] X.-M. Zhang, T. Li, and X. Yuan, “Quantum state preparation with
optimal circuit depth: Implementations and applications,” Phys. Rev.
Lett. , vol. 129, no. 23, Nov. 2022, Art. no. 230504, doi:10.1103/Phys-
RevLett.129.230504.
[22] W. Li and D.-L. Deng, “Recent advances for quantum classifiers,”
Sci. China Phys., Mechanics Astron. , vol. 65, no. 2, Dec. 2021,
Art. no. 220301, doi:10.1007/s11433-021-1793-6.
[23] N. Moustafa and J. Slay, “UNSW-NB15: A comprehensive data set for
network intrusion detection systems (UNSW-NB15 network data set),”
in Proc. Mil. Commun. Inf. Syst. Conf. , 2015, pp. 1–6, doi:10.1109/Mil-
CIS.2015.7348942.
[24] N. Moustafa, G. Creech, and J. Slay, “Big Data analytics for intrusion
detection system: Statistical decision-making using finite Dirichlet mix-
ture models,” in Data Analytics and Decision Support for Cybersecurity:
Trends, Methodologies and Applications, ser. Data Analytics , I. Palomares
Carrascosa, H. K. Kalutarage, and Y. Huang, Eds., Berlin, Germany:
Springer, 2017, pp. 127–156, doi:10.1007/978-3-319-59439-2_5.
[25] N. Moustafa and J. Slay, “The evaluation of network anomaly detection
systems: Statistical analysis of the UNSW-NB15 data set and the compar-
ison with the KDD99 data set,” Inf. Secur. J., A Glob. Perspective , vol. 25,
no. 1-3, pp. 18–31, Apr. 2016, doi:10.1080/19393555.2015.1124946.
[26] N. Moustafa, J. Slay, and G. Creech, “Novel geometric area analysis
technique for anomaly detection using trapezoidal area estimation on
large-scale networks,” IEEE Trans. Big Data , vol. 5, no. 4, pp. 481–494,
Dec. 2019, doi:10.1109/TBDATA.2017.2715166.
[27] Y. Yang, K. Zheng, B. Wu, Y. Yang, and X. Wang, “Network intru-
sion detection based on supervised adversarial variational auto-encoder
with regularization,” IEEE Access , vol. 8, pp. 42169–42184, 2020,
doi:10.1109/ACCESS.2020.2977007.
[28] M. Shahin, F. F. Chen, H. Bouzary, A. Hosseinzadeh, and R. Rashidifar,
“A novel fully convolutional neural network approach for detection and
classification of attacks on Industrial IoT devices in smart manufacturing
systems,” Int. J. Adv. Manuf. Technol. , vol. 123, no. 5, pp. 2017–2029,
Nov. 2022, doi:10.1007/s00170-022-10259-3.
[29] J. B. Awotunde, C. Chakraborty, and A. E. Adeniyi, “Intrusion detection in
Industrial Internet of Things network-based on deep learning model with
rule-based feature selection,” Wirel. Commun. Mobile Comput. , vol. 2021,
Sep. 2021, Art. no. e7154587, doi:10.1155/2021/7154587.
[30] J. Gu and S. Lu, “An effective intrusion detection approach using SVM
with naïve Bayes feature embedding,” Comput. Secur. , vol. 103, Apr. 2021,
Art. no. 102158, doi:10.1016/j.cose.2020.102158.
```

## Kukliansky et al.: NETWORK ANOMALY DETECTION USING QUANTUM NEURAL NETWORKS uantumEngineering

[31] P. Jayalaxmi, G. Kumar, R. Saha, M. Conti, T.-H. Kim, and R. Thomas,
“Debot: A deep learning-based model for bot detection in Industrial
Internet-of-Things,” _Comput. Elect. Eng._ , vol. 102, 2022, Art. no. 108214,
doi:10.1016/j.compeleceng.2022.108214.
[32] M. Vishwakarma and N. Kesswani, “DIDS: A deep neural network based
real-time intrusion detection system for IoT,” _Decis. Analytics J._ ,vol.5,
Dec. 2022, Art. no. 100142, doi:10.1016/j.dajour.2022.100142.
[33] P. Russell, M. A. Elsayed, B. Nandy, N. Seddigh, and N. Zincir-
Heywood, “On the fence: Anomaly detection in IoT networks,” in
_Proc. IEEE/IFIP Netw. Operations Manage. Symp._ , 2023, pp. 1–4,
doi:10.1109/NOMS56928.2023.10154271.
[34] A. Moore, D. Zuev, and M. Crogan, “Discriminators for use in flow-based
classification,” Dept. Comput. Sci., Queen Mary Univ. London, London,
U.K., Tech. Rep., RR-05-13, 2013. [Online]. Available: https://
qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/5050/RR-05-13.pdf
[35] A. Shiravi, H. Shiravi, M. Tavallaee, and A. A. Ghorbani, “Toward de-
veloping a systematic approach to generate benchmark datasets for intru-
sion detection,” _Comput. Secur._ , vol. 31, no. 3, pp. 357–374, May 2012,
doi:10.1016/j.cose.2011.12.012.
[36] “KDD Cup 1999 Data,” [Online]. Available: [http://kdd.ics.uci.edu/](http://kdd.ics.uci.edu/)
databases/kddcup99/kddcup99.html
[37] M. Sarhan, S. Layeghy, and M. Portmann, “Feature analysis
for machine learning-based IoT intrusion detection,” 2022,
doi:10.48550/arXiv.2108.12732.
[38] “MNIST classification | TensorFlow Quantum,” [Online]. Available:
https://www.tensorflow.org/quantum/tutorials/mnist
[39] G. Vidal and C. M. Dawson, “Universal quantum circuit for two-qubit
transformations with three controlled-NOT gates,” _Phys. Rev. A_ , vol. 69,
no. 1, Jan. 2004, Art. no. 010301, doi:10.1103/PhysRevA.69.010301.
[40] M. Broughton et al., “TensorFlow quantum: A software framework for
quantum machine learning,” 2021, doi:10.48550/arXiv.2003.02989.
[41] C. Gentile and M. K. K. Warmuth, “Linear Hinge Loss and Average
Margin,” in _Advances in Neural Information Processing Systems_ , vol. 11.
Cambridge, MA, USA: MIT Press, 1998, doi:10.5555/3009055.3009087.
[42] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
2017, doi:10.48550/arXiv.1412.6980.
[43] “IonQ Harmony,” [Online]. Available: https://ionq.com/quantum-
systems/harmony
[44] “Debiasing and sharpening,” [Online]. Available: https://ionq.com/
resources/debiasing-and-sharpening
[45] A. Maksymov, J. Nguyen, Y. Nam, and I. Markov, “Enhancing
quantum computer performance via symmetrization,” 2023,
doi:10.48550/arXiv.2301.07233.
[46] “IonQ Aria,” [Online]. Available: https://ionq.com/quantum-systems/aria

```
Alon Kuklianskyreceived the B.Sc degree in
electrical and computer engineering and the M.Sc
degree in electrical engineering from Tel-Aviv
University, Tel Aviv, Israel, in 2008 and 2016,
respectively. He is currently working toward the
Ph.D. degree in quantum computing with the De-
partment of Electrical and Computer Engineer-
ing, Naval Postgraduate School, Monterey, CA,
USA.
His research interests include quantum com-
puting and computer architectures.
```
```
Marko Orescanin(Member, IEEE) received
the Ph.D. degree in electrical and computer en-
gineering from the University of Illinois Urbana-
Champaign, Champaign, IL, USA, in 2010.
From 2011 to 2019, he was with Bose Cor-
poration, MA, USA, where he primarily worked
on research and advanced development of signal
processing and machine learning algorithms for
audio and speech enhancement in consumer elec-
tronics. He left Bose as a Senior Manager of AI
and Data group with focus on consumer electron-
ics business unit. Since 2019, he has been an Assistant Professor with the
Computer Science Department, Naval Postgraduate School, Monterey, CA,
USA. His research interests include signal processing, machine learning,
artificial intelligence, Bayesian deep learning, and cyber-physical system
security.
```
```
Chad Bollmann(Senior Member, IEEE) re-
ceived the B.S. degree in ocean engineering from
the U.S. Naval Academy, Annapolis, MD, USA,
in 1996, the S.M.s degree in technology and
policy and nuclear engineering from the Mas-
sachusetts Institute of Technology, Cambridge,
MA, USA, in 1998, and Ph.D. degree in electri-
cal engineering from Naval Postgraduate School
(NPS), Monterey, CA, USA, in 2018.
After entering the U.S. Navy in 1992, from
1998–2014 he was the Submarine Warfare Offi-
cer, most recently as the Executive Officer of the USS NEVADA (GOLD)
(SSBN-733). He is currently working as an Assistant Professor of electrical
and computer engineering, NPS. He is currently a Captain in the US Navy
and co-appointed as a Permanent Military Professor. He is the Director of the
NPS Center for Cyber Warfare. His teaching and research interests include
cyber-physical system security, network traffic analysis, and non-Gaussian
statistical signal processing.
```
```
Theodore Huffmirereceived the A.B. degree
in computer science from Princeton University,
Princeton, NJ, USA, in 1997, and the Ph.D. de-
gree in computer science from the University of
California, Santa Barbara, Santa Barbara, CA,
USA, in 2007.
He is currently an Associate Professor with the
Department of Computer Science, Naval Post-
graduate School, Monterey, CA, USA. His re-
search interests include intersection of computer
architecture and computer security.
Dr. Huffmire’s was the recipient of the Science, Mathematics, and Re-
search for Transformation Scholarship (U.S. Department of Defense) and
the Navy Meritorious Civilian Service Award (U.S. Navy).
```

