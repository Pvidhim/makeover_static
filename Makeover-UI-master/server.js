// server.js
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json()); // For parsing application/json

// Example endpoint to call the Python API
app.get('/api/data', async (req, res) => {
    try {
        const response = await axios.get('http://your-python-api-url/data'); // Replace with your Python API endpoint
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching data from Python API:', error);
        res.status(500).send('Error fetching data from Python API');
    }
});

// Serve static files from the React app
app.use(express.static('build'));

// Serve the React app
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://192.168.1.24:${PORT}`);
});
