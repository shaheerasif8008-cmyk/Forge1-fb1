#!/usr/bin/env python3
"""
Test OpenAI Integration for Employee Memory System

This script tests the OpenAI API integration for generating embeddings
used in the AI Employee Memory Management system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the forge1 backend to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from forge1.services.employee_memory_manager import EmployeeMemoryManager


async def test_openai_embeddings():
    """Test OpenAI embeddings generation"""
    print("🧠 Testing OpenAI Embeddings Integration...")
    
    # Initialize memory manager
    memory_manager = EmployeeMemoryManager()
    await memory_manager.initialize()
    
    # Test embedding generation
    test_texts = [
        "Hello, I am a corporate lawyer specializing in mergers and acquisitions.",
        "Can you help me review this contract for potential legal issues?",
        "What are the key considerations for due diligence in M&A transactions?",
        "I need assistance with drafting a non-disclosure agreement.",
        "Please analyze the liability clauses in this service agreement."
    ]
    
    print(f"\n📊 Generating embeddings for {len(test_texts)} test texts...")
    
    embeddings = []
    for i, text in enumerate(test_texts, 1):
        print(f"  {i}. Generating embedding for: {text[:50]}...")
        
        try:
            embedding = await memory_manager._generate_embedding(text)
            embeddings.append(embedding)
            
            print(f"     ✅ Success! Embedding dimension: {len(embedding)}")
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            return False
    
    # Test similarity calculation
    print(f"\n🔍 Testing embedding similarity...")
    
    if len(embeddings) >= 2:
        # Calculate cosine similarity between first two embeddings
        import numpy as np
        
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            print(f"  Similarity between text 1 and 2: {similarity:.4f}")
        
        # Test with identical text (should be very similar)
        embedding_duplicate = await memory_manager._generate_embedding(test_texts[0])
        vec_dup = np.array(embedding_duplicate)
        
        similarity_identical = np.dot(vec1, vec_dup) / (np.linalg.norm(vec1) * np.linalg.norm(vec_dup))
        print(f"  Similarity with identical text: {similarity_identical:.4f}")
        
        if similarity_identical > 0.99:
            print("  ✅ Identical text similarity test passed!")
        else:
            print("  ⚠️  Identical text similarity lower than expected")
    
    print(f"\n✅ OpenAI embeddings integration test completed successfully!")
    print(f"   - Model: {memory_manager.embedding_model}")
    print(f"   - Dimensions: {memory_manager.embedding_dimensions}")
    print(f"   - Cache size: {len(memory_manager._embedding_cache)}")
    
    return True


async def test_memory_operations():
    """Test basic memory operations"""
    print(f"\n🧪 Testing Memory Operations...")
    
    # This would require a database connection, so we'll just test the setup
    memory_manager = EmployeeMemoryManager()
    await memory_manager.initialize()
    
    # Test namespace generation
    namespace = memory_manager._get_namespace("client_123", "emp_456")
    expected_namespace = "client_123:emp_456"
    
    if namespace == expected_namespace:
        print(f"  ✅ Namespace generation: {namespace}")
    else:
        print(f"  ❌ Namespace generation failed: got {namespace}, expected {expected_namespace}")
        return False
    
    print(f"  ✅ Memory operations setup test passed!")
    return True


async def main():
    """Main test function"""
    print("🚀 Starting Forge1 OpenAI Integration Tests...\n")
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in the .env file")
        return
    
    if openai_key.startswith("sk-"):
        print(f"✅ OpenAI API key found: {openai_key[:20]}...")
    else:
        print("⚠️  OpenAI API key format looks unusual")
    
    # Run tests
    try:
        # Test embeddings
        embeddings_success = await test_openai_embeddings()
        
        # Test memory operations
        memory_success = await test_memory_operations()
        
        if embeddings_success and memory_success:
            print(f"\n🎉 All tests passed! OpenAI integration is working correctly.")
            print(f"\n📋 Next steps:")
            print(f"   1. Run the database migrations to set up the schema")
            print(f"   2. Start the Forge1 backend server")
            print(f"   3. Test the complete employee lifecycle APIs")
        else:
            print(f"\n❌ Some tests failed. Please check the configuration.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())