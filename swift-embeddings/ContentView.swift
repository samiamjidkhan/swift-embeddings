//
//  ContentView.swift
//  swift-embeddings
//
//  Created by Sami Khan on 2024-12-31.
//

import SwiftUI
import Embeddings

struct ContentView: View {
    @State private var inputText: String = ""
    @State private var embedding: [Float]?
    @State private var isLoading: Bool = false
    @State private var error: String?
    
    private let generator = EmbeddingGenerator()
    
    var body: some View {
        VStack(spacing: 20) {
            TextField("Enter text to embed", text: $inputText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            Button(action: generateEmbedding) {
                Text(isLoading ? "Processing..." : "Generate Embedding")
            }
            .disabled(inputText.isEmpty || isLoading)
            
            if let error = error {
                Text(error)
                    .foregroundColor(.red)
            }
            
            if let embedding = embedding {
                Text("Embedding (first 5 values):")
                Text(String(describing: Array(embedding.prefix(5))))
                    .font(.system(.body, design: .monospaced))
            }
        }
        .padding()
        .frame(minWidth: 400, minHeight: 300)
        .task {
            do {
                try await generator.initialize()
            } catch {
                self.error = "Failed to initialize model: \(error.localizedDescription)"
            }
        }
    }
    
    private func generateEmbedding() {
        isLoading = true
        error = nil
        
        Task {
            do {
                let result = try await generator.getEmbedding(for: inputText)
                embedding = result
            } catch {
                self.error = "Error: \(error.localizedDescription)"
            }
            isLoading = false
        }
    }
}

#Preview {
    ContentView()
}

class EmbeddingGenerator {
    private var modelBundle: Bert.ModelBundle?
    
    func initialize() async throws {
        // Get the main bundle's resource URL
        let resourceURL = Bundle.main.resourceURL!
        print("Debug: Using resource URL:", resourceURL.path)
        
        // Verify all required files exist (including additional tokenizer files)
        let requiredFiles = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
        
        for file in requiredFiles {
            let filePath = resourceURL.appendingPathComponent(file).path
            let exists = FileManager.default.fileExists(atPath: filePath)
            print("Debug: Checking \(file): \(exists)")
            guard exists else {
                throw NSError(domain: "EmbeddingGenerator", code: 3, userInfo: [
                    NSLocalizedDescriptionKey: "Missing required model file: \(file)"
                ])
            }
        }
        
        // Load the model from the resources directory
        do {
            modelBundle = try await Bert.loadModelBundle(from: resourceURL)
        } catch {
            print("Debug: Model loading error details:", error)
            throw error
        }
    }
    
    func getEmbedding(for text: String) async throws -> [Float] {
        guard let modelBundle = modelBundle else {
            throw NSError(domain: "EmbeddingGenerator", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Model not initialized"
            ])
        }
        
        // Encode the text and convert to Float array
        let encoded = try modelBundle.encode(text)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        return Array(result)
    }
}

// Usage example:
/*
let generator = EmbeddingGenerator()
Task {
    do {
        try await generator.initialize()
        let embedding = try await generator.getEmbedding(for: "The cat is black")
        print(embedding)
    } catch {
        print("Error: \(error)")
    }
}
*/

