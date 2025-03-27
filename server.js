const express = require("express");
const axios = require("axios");
const cors = require("cors");

const app = express();
const PORT = 3000;

// Allow frontend to access server
app.use(cors());
app.use(express.static("public"));

const THINGSPEAK_API_KEY = "YLFVDZN8HIM7850I";
const THINGSPEAK_CHANNEL_ID = "2855545";  // Replace with your ThingSpeak Channel ID
const THINGSPEAK_URL = `https://api.thingspeak.com/channels/${THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key=${THINGSPEAK_API_KEY}`;

app.get("/data", async (req, res) => {
    try {
        const response = await axios.get(THINGSPEAK_URL);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: "Failed to fetch data" });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
