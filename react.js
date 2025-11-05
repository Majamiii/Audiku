import React, { useState } from 'react';
import { Upload, Music, FileAudio, Loader2, Sparkles } from 'lucide-react';

export default function AudioHaikuGenerator() {
  const [audioFile, setAudioFile] = useState(null);
  const [audioPath, setAudioPath] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [description, setDescription] = useState('');
  const [haiku, setHaiku] = useState('');
  const [error, setError] = useState('');
  const [stage, setStage] = useState(''); // 'analyzing', 'generating', 'complete'

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioPath(file.name);
      setDescription('');
      setHaiku('');
      setError('');
    }
  };

  const handlePathChange = (e) => {
    setAudioPath(e.target.value);
    setAudioFile(null);
    setDescription('');
    setHaiku('');
    setError('');
  };

  const processAudio = async () => {
    if (!audioFile && !audioPath) {
      setError('Please select an audio file or enter a file path');
      return;
    }

    setIsProcessing(true);
    setError('');
    setStage('analyzing');

    try {
      let response;
      
      if (audioFile) {
        // Upload file
        const formData = new FormData();
        formData.append('audio', audioFile);
        
        response = await fetch('http://localhost:5000/api/generate-haiku', {
          method: 'POST',
          body: formData,
        });
      } else {
        // Send file path
        response = await fetch('http://localhost:5000/api/generate-haiku', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ path: audioPath }),
        });
      }

      if (!response.ok) {
        throw new Error('Failed to process audio');
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setDescription(data.description);
      setStage('generating');
      
      // Small delay to show the generating stage
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setHaiku(data.haiku);
      setStage('complete');
    } catch (err) {
      setError(`Error: ${err.message}. Please ensure Python backend is running on port 5000.`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Music className="w-10 h-10 text-purple-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              Audio Haiku Generator
            </h1>
          </div>
          <p className="text-gray-600">Transform audio into sentimental poetry</p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          {/* File Upload Section */}
          <div className="mb-6">
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Upload Audio File
            </label>
            <div className="relative">
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
                id="audio-upload"
              />
              <label
                htmlFor="audio-upload"
                className="flex items-center justify-center gap-3 w-full p-6 border-2 border-dashed border-purple-300 rounded-xl hover:border-purple-500 hover:bg-purple-50 transition-all cursor-pointer"
              >
                <Upload className="w-6 h-6 text-purple-500" />
                <span className="text-gray-600">
                  {audioFile ? audioFile.name : 'Click to upload audio file'}
                </span>
              </label>
            </div>
          </div>

          {/* Or Divider */}
          <div className="flex items-center gap-4 mb-6">
            <div className="flex-1 h-px bg-gray-300"></div>
            <span className="text-gray-500 text-sm font-medium">OR</span>
            <div className="flex-1 h-px bg-gray-300"></div>
          </div>

          {/* File Path Input */}
          <div className="mb-6">
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Enter Audio File Path
            </label>
            <div className="relative">
              <FileAudio className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={audioPath}
                onChange={handlePathChange}
                placeholder="./audio_samples/rock.mp3"
                className="w-full pl-12 pr-4 py-3 border-2 border-gray-300 rounded-xl focus:border-purple-500 focus:outline-none transition-colors"
              />
            </div>
          </div>

          {/* Generate Button */}
          <button
            onClick={processAudio}
            disabled={isProcessing || (!audioFile && !audioPath)}
            className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-3 shadow-lg hover:shadow-xl"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                {stage === 'analyzing' && 'Analyzing Audio...'}
                {stage === 'generating' && 'Generating Haiku...'}
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Generate Haiku
              </>
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        {(description || haiku) && (
          <div className="space-y-6">
            {/* Audio Description */}
            {description && (
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <FileAudio className="w-5 h-5 text-purple-600" />
                  Audio Analysis
                </h2>
                <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono bg-gray-50 p-4 rounded-xl">
                  {description}
                </pre>
              </div>
            )}

            {/* Generated Haiku */}
            {haiku && (
              <div className="bg-gradient-to-br from-purple-100 via-pink-100 to-blue-100 rounded-2xl shadow-xl p-8">
                <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-purple-600" />
                  Your Haiku
                </h2>
                <div className="bg-white/80 backdrop-blur rounded-xl p-8 text-center">
                  <p className="text-2xl text-gray-800 leading-relaxed font-serif whitespace-pre-line">
                    {haiku}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Info Card */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
          <h3 className="font-semibold text-blue-900 mb-2">Note</h3>
          <p className="text-blue-800 text-sm">
            This is a frontend interface. To use it with your Python backend:
            <br />• Run a Flask/FastAPI server with your audio processing code
            <br />• Update the API endpoints in the code
            <br />• Ensure all dependencies are installed (whisper, tensorflow, ollama, etc.)
          </p>
        </div>
      </div>
    </div>
  );
}