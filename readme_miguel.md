# Interpreting Large Language Models Using Sparse Autoencoders

## Introduction

The following report explains a method for interpreting Large Language Models (LLMs) using Sparse Autoencoders (SAEs). The core idea is to attach a sparse autoencoder to a specific layer of an LLM in order to identify neurons that activate (i.e., take non-zero values) for particular tokens or concepts. By doing so, we aim to reveal which neurons correspond to abstract concepts represented in the model.

In essence, this approach enables the transformation of complex, entangled representations within LLMs into more interpretable components, where individual neurons correspond more directly to specific semantic features.

## Autoencoders

An autoencoder is a neural network that takes an input, processes it through a hidden layer, and then uses a decoder to reconstruct the original input. The objective is to learn a representation such that the output closely matches the input.

Thus, an autoencoder learns to reconstruct its input from a latent (hidden) representation. The key insight is that the hidden layer captures meaningful structure from the data.

- If the hidden layer has fewer neurons than the input layer, the model performs compression, producing a lower-dimensional representation of the input.
- If the hidden layer has more neurons than the input layer, the model performs expansion, creating a higher-dimensional representation of the input.

## Sparse Autoencoders (SAE)

A Sparse Autoencoder is a type of autoencoder in which the hidden layer has more neurons than the input, but sparsity is enforced through a penalty in the loss function. This ensures that most neurons in the hidden layer remain inactive (i.e., output zero) for any given input.

Sparsity is typically enforced using L1 regularization.

The loss function is defined as:

$$
\\text{Loss} = \\text{Reconstruction Loss} + \\lambda \\times \\text{L1 Loss}
$$

Where:

- Reconstruction Loss is the mean squared error (MSE) between the original and reconstructed activations. The reconstruction loss is computed using Mean Squared Error (MSE), as the model is trained to reconstruct continuous activation vectors rather than predict discrete classes.
- L1 Loss is the mean absolute value of latent activations
- $\\lambda$ (lambda) controls the strength of sparsity

### Purpose

- Reconstruction loss ensures that the model preserves information
- L1 loss encourages sparsity and improves interpretability

As a result, Sparse Autoencoders produce a higher-dimensional but less entangled representation, where features are more separable.

The regularization parameter $\\lambda$ controls the strength of the L1 penalty and therefore the level of sparsity in the latent representation. A higher $\\lambda$ enforces greater sparsity, meaning fewer neurons activate for any given input, while a lower $\\lambda$ allows more neurons to remain active.

If $\\lambda$ is too high, the model may become overly sparse, forcing most neurons to zero. This can lead to loss of important information, poor reconstruction of activations, and degraded interpretability, as meaningful features may be suppressed.

If $\\lambda$ is too low, the representation becomes dense, with many neurons active simultaneously. This reduces sparsity and leads to entangled features, making it harder to interpret individual neurons and limiting the benefits of the Sparse Autoencoder.

In practice, $\\lambda$ must be carefully tuned to balance information preservation and interpretability, achieving a representation that is both accurate and sufficiently sparse.

## Sparse Autoencoder Loss Function

$$
\\mathcal{L} = \\frac{1}{n}\\sum\_{i=1}^{n}\\lVert x_i - \\hat{x}_i \\rVert_2^2 + \\lambda \\cdot \\frac{1}{n}\\sum_{i=1}^{n}\\lVert z_i \\rVert_1
$$

Where:

- $x_i$ = original activation vector
- $\\hat{x}\_i$ = reconstructed activation vector
- $z_i$ = latent representation (SAE hidden layer activations)
- $n$ = number of samples
- $\\lambda \\in \\mathbb{R}^{+}$ = regularization parameter controlling sparsity

## Applying Sparse Autoencoders to LLMs

In this work, a Sparse Autoencoder is attached to a specific layer of the LLM. The input to the SAE consists of the activations (embeddings) produced by the LLM up to that layer, and the output is a reconstruction of those activations.

The key insight is that by analyzing the hidden (latent) layer of the SAE, we can isolate interpretable features (concepts). Each neuron in this latent space represents a feature, and its activation indicates the presence of that feature in the input.

Additionally, once these features are identified, they can be manipulated (clamped) to influence the behavior of the LLM.

## From Polysemanticity to Monosemanticity

LLMs often exhibit polysemanticity, where a single neuron represents multiple unrelated concepts. This occurs due to superposition, where the model uses nearly orthogonal vectors to encode more features than available dimensions.

The use of Sparse Autoencoders allows us to transform these polysemantic representations into monosemantic ones, where each neuron corresponds to a single interpretable feature.

## Core Idea and Methodology

The overall process for interpreting LLMs using Sparse Autoencoders can be summarized as follows:

1. Pass many prompts through the LLM and capture hidden activations from a specific layer
1. Learn a sparse and interpretable representation of these activations using a Sparse Autoencoder
1. Map the resulting latent features (neurons) back to the tokens and their context
1. Analyze these mappings to understand what each feature represents
1. Perform interventions by clamping selected features to test their causal effect on the model’s behavior

## LLM Model Description

The LLM model used in this work is:

- Model: Qwen2.5-0.5B-Instruct
- Type: Transformer-based causal language model
- Size: Approximately 0.5 billion parameters
- Layers: 24
- Architecture: Decoder-only (autoregressive)

The model is autoregressive, meaning it predicts the next token given previous tokens and feeds that prediction back into the model iteratively:

$$
\\text{prompt} \\rightarrow \\text{predict token} \\rightarrow \\text{append} \\rightarrow \\text{repeat}
$$

## Where the SAE is Applied

Activations are extracted from the middle residual stream layer of the transformer. Since the model has 24 layers, the SAE is applied at layer 13.

- Early layers capture low-level patterns
- Late layers are highly specialized toward final predictions
- Middle layers provide the best balance for interpretability

The selected layer produces activation vectors of size 896.

## Limitations of the Selected Interpretability Approach

While applying a Sparse Autoencoder (SAE) to a middle residual stream layer provides valuable insights into LLM behavior, this approach has several limitations that prevent it from fully explaining the model.

First, the method provides only a partial view of the model’s internal processes. By focusing on a single layer (layer 13), the analysis ignores interactions across other layers, even though LLM behavior emerges from complex, multi-layer computations.

Second, although sparsity encourages interpretability, features are not guaranteed to be perfectly monosemantic. Some neurons may still represent multiple overlapping concepts, limiting the clarity of interpretations.

Finally, the approach is data-dependent, meaning that the learned features reflect only the distribution of prompts used during training. As a result, interpretations may not generalize to unseen domains or different types of inputs.

In summary, while this method improves transparency and provides meaningful insights, it does not offer a complete or definitive explanation of LLM behavior.

## SAE Architecture

The Sparse Autoencoder is configured as follows:

- Input dimension: 896
- Latent dimension: 1792
- Activation function: ReLU
- Decoder: tied weights

### Architecture

$$
\\text{Input (896)} \\rightarrow \\text{Encoder} \\rightarrow \\text{Latent (1792)} \\rightarrow \\text{Decoder} \\rightarrow \\text{Reconstruction (896)}
$$

Each neuron in the latent space represents a feature, meaning:

$$
1792\\ \\text{neurons} = 1792\\ \\text{features}
$$

Activation space refers to the vector space formed by the internal activations of a neural network at a given layer (896 neurons in this case). The number of neurons defines the dimension of the activation space, but the activation space itself is the set of all possible activation vectors those neurons can produce. Activation space and feature space are not the same; rather, the feature space (1792 neurons in this case) is a transformed version of the activation space, where the Sparse Autoencoder reorganizes the original dense and entangled representations into a sparse and more interpretable set of features.

## Training the SAE

The SAE is trained using activations extracted from the 13th layer of the LLM, obtained by passing a large corpus of prompts through the LLM.

### Process

1. Define a prompt corpus (50 soccer-related prompts)
1. Tokenize each prompt
1. Pass prompts through the LLM
1. Capture hidden activations
1. Extract activation vectors per token

Each token generates one activation vector, which is then used as input to the SAE.

### Dataset Shape

$$
\[\\text{num_tokens}, \\text{hidden_dim}\]
$$

Example:

$$
\[467, 896\]
$$

## Tokenization and Activation Mapping

Text is first converted into tokens using the LLM tokenizer.

Example:

"What is Messi?" → \["What", " is", " Me", "ssi", "?"\] (illustrative example)

Each token corresponds to a single activation vector extracted from the LLM.

## Mapping Features to Tokens and Context

The mapping process follows:

$$
\\text{Token} \\rightarrow \\text{Activation} \\rightarrow \\text{Latent Features}
$$

### Steps

1. Each token has an activation vector
1. The SAE encodes it into latent features
1. For each feature (latent neuron):
   - Rank the top tokens that activate a given feature the most by sorting latent activations and selecting the highest values
   - Retrieve the corresponding tokens
   - Retrieve the context (the full prompt where the token appears)
   - Interpretation is performed by analyzing: **"What do the top activating tokens have in common?"**

## Feature Interpretation

Examples of learned features include:

- Feature 0 → Question-related tokens ("What", "Why")
- Feature 10 → Soccer player names (e.g., Messi, Ronaldo)

These examples demonstrate how latent neurons correspond to meaningful semantic concepts.

## Clamping (Intervention and Causality)

Clamping is used to test causal relationships between features and model outputs.

### Definition

A latent feature (SAE neuron) is forced to a specific value during the forward pass to perform counterfactual experiments.

The clamping rule is:

$$
\\text{feature value} = \\text{multiplier} \\times \\max(\\text{feature})
$$

Where:

- multiplier is a chosen constant (e.g., 10)
- $\\max(\\text{feature})$ is the maximum observed activation of that feature

This ensures that the intervention is scale-aware.

Because the LLM is autoregressive:

- It generates one token at a time
- Each token generation triggers a forward pass

Therefore, clamping must be applied at every generation step.

Clamping alters the model’s behavior.

Example:

Clamping a “player name” feature results in:

- Increased repetition of player names
- More entity-focused outputs
- Less structured explanations

This demonstrates that the feature has a causal influence on the model’s behavior.

## Conclusion

By attaching a Sparse Autoencoder to an intermediate layer of an LLM, we can extract interpretable features that correspond to meaningful concepts. This approach transforms polysemantic representations into more interpretable, monosemantic ones.

Furthermore, by clamping these features, we can experimentally verify their causal impact on model outputs, providing a powerful framework for understanding and controlling LLM behavior.

## Useful Definitions

- **Latent Space:**
  Latent space refers to the vector space formed by the activations of the hidden layer of the Sparse Autoencoder (SAE). The number of neurons defines the dimensionality of this space, but the latent space itself consists of all possible activation vectors that these neurons can produce. It represents a transformed version of the original activation space, where information is encoded in a sparse and more interpretable form, with each dimension corresponding to a learned feature.

- **Feature:**
  A feature corresponds to a single neuron in the latent (hidden) layer of the Sparse Autoencoder. Each feature represents a specific pattern or concept learned from the input data, and its activation indicates the presence of that concept in a given token or context.

- **Activations:**
  Activations refer to the numerical outputs of neurons after applying a transformation (e.g., linear operation followed by a non-linear activation function). In the context of a neural network layer, activations are typically represented as a vector, where each value corresponds to the output of a neuron for a given input.

- **Norm ($|\\cdot|$):**
  A norm measures the magnitude of a vector. In this work:

$$
\\lVert x_i - \\hat{x}_i \\rVert_2^2 = \\sum_{j=1}^{d}(x\_{i,j} - \\hat{x}\_{i,j})^2
$$

$$
\\lVert z_i \\rVert_1 = \\sum\_{j=1}^{k} |z\_{i,j}|
$$

$$
\\lVert z_i \\rVert_2 = \\sqrt{\\sum\_{j=1}^{k} z\_{i,j}^2}
$$

## References

- Scaling Monosemanticity: Extracting interpretable features from Claude 3 Sonnet. (2024). Anthropic.

## File Descriptions

### `src/`

- **model_wrapper.py** – Loads the LLM and provides utilities to run prompts and extract hidden activations from a specific layer.
- **data_generator.py** – Generates the SAE training dataset by passing prompts through the LLM and saving token-level activations with metadata.
- **sae.py** – Defines the Sparse Autoencoder architecture (encoder, decoder, and loss function).
- **trainer.py** – Implements the training loop for the SAE, including optimization and loss tracking.
- **interpretation.py** – Provides tools to analyze latent features and map them back to tokens and their context.
- **intervention.py** – Implements feature clamping (intervention) by modifying latent features during the LLM forward pass.

### `scripts/`

- **generate_dataset.py** – Runs the full pipeline to create and save the activation dataset from a set of prompts.
- **train_sae.py** – Trains the Sparse Autoencoder using the generated activation dataset.
- **inspect_features.py** – Allows selecting specific features to inspect and analyze their corresponding tokens and contexts.
- **clamp_feature.py** – Performs the steering (intervention) analysis by clamping a feature and comparing baseline vs modified outputs.
