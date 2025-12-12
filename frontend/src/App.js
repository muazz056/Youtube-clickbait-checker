import React, { useState } from 'react';
import { Container, TextField, Button, Typography, Box, Paper, CircularProgress } from '@mui/material';
import axios from 'axios';

function App() {
  const [link, setLink] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    try {
      // Step 1: Extract video data
      const extractRes = await axios.post('http://localhost:8000/extract_video_data', { url: link });
      const videoData = extractRes.data;
      if (!videoData.success) {
        setError(videoData.error || 'Failed to extract video data');
        setLoading(false);
        return;
      }
      // Step 2: Predict clickbait
      const predictRes = await axios.post('http://localhost:8000/predict', {
        title: videoData.title,
        description: videoData.description,
        tags: videoData.tags,
        thumbnail_text: videoData.thumbnail_text,
        transcript: videoData.transcript
      });
      setResult({ ...videoData, ...predictRes.data });
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Clickbait Checker</Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            label="Video Link"
            fullWidth
            value={link}
            onChange={e => setLink(e.target.value)}
            margin="normal"
            required
          />
          <Button type="submit" variant="contained" color="primary" disabled={loading} fullWidth>
            {loading ? <CircularProgress size={24} /> : 'Check'}
          </Button>
        </form>
        {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>}
        {result && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6">Result:</Typography>
            <Typography><b>Title:</b> {result.title}</Typography>
            <Typography><b>Description:</b> {result.description}</Typography>
            <Typography><b>Tags:</b> {result.tags?.join(', ')}</Typography>
            <Typography><b>Thumbnail Text:</b> {result.thumbnail_text}</Typography>
            <Typography><b>Scene Description:</b> {result.scene_description}</Typography>
            <Typography><b>Transcript:</b> {result.transcript}</Typography>
            {result.transcript_source && (
              <Typography variant="body2" color="textSecondary">
                Transcript source: <b>{
                  result.transcript_source === 'manual' ? 'Manual captions' :
                  result.transcript_source === 'auto' ? 'Auto-generated captions' :
                  result.transcript_source === 'whisper' ? 'Audio (Whisper)' : result.transcript_source
                }</b>
              </Typography>
            )}
            <Box sx={{ mt: 2 }}>
              <Typography variant="h5">
                Clickbait: <b>{result.clickbait}</b> {result.confidence !== undefined ? `(Confidence: ${(result.confidence * 100).toFixed(1)}%)` : ''}
              </Typography>
            </Box>
            {result.thumbnail && <img src={result.thumbnail} alt="Thumbnail" style={{ width: '100%', marginTop: 16 }} />}
          </Box>
        )}
      </Paper>
    </Container>
  );
}

export default App; 