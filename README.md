# Swift Embeddings

A macOS application that generates text embeddings using BERT, built with SwiftUI and the Apple Embeddings framework.

## Overview

Swift Embeddings is a simple desktop application that allows users to generate vector embeddings from text input. It uses Apple's Embeddings framework to run BERT (Bidirectional Encoder Representations from Transformers) locally on your machine.

## Features

- Clean, native macOS interface
- Real-time text embedding generation
- Display of embedding vector preview (first 5 values)
- Error handling and loading states
- Sandboxed application for security

## Requirements

- macOS 14.0 or later
- Xcode 15.0 or later

## Installation

1. Clone the repository
2. Make sure BERT model files are placed them in the app's resource directory:
   - config.json
   - pytorch_model.bin
   - tokenizer.json
   - vocab.txt
   - tokenizer_config.json
   - special_tokens_map.json
3. Open the project in Xcode
4. Build and run the application

## Usage

1. Launch the application
2. Enter text in the input field
3. Click "Generate Embedding" to create the embedding vector
4. View the first 5 values of the resulting embedding

## Technical Details

- Built with SwiftUI
- Uses Apple's Embeddings framework for BERT model integration
- Implements async/await for non-blocking operations
- Includes comprehensive error handling and debug logging
