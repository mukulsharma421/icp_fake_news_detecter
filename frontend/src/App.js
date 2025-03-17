import { useState } from "react";
import './App.css'; // Make sure to add this line to include the spinner style

function App() {
  const [textInput, setTextInput] = useState("");
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = async () => {
    setPrediction("");
    setError(null);

    if (!textInput.trim()) {
      setError("Please enter some text.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ news: textInput }), 
      });

      if (!response.ok) {
        throw new Error("Error with the prediction request. Please try again.");
      }

      const data = await response.json();
      if (data.prediction) {
        setPrediction(data.prediction);
      } else {
        setError("Prediction could not be generated. Please try again.");
      }
    } catch (err) {
      setError(err.message || "An error occurred while making the prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Fake News Prediction</h1>
      <textarea
        aria-label="Enter your news article here"
        value={textInput}
        onChange={(e) => setTextInput(e.target.value)}
        placeholder="Enter your news article here"
        rows="6"
        cols="50"
        disabled={loading}
        style={{ width: "80%", marginBottom: "15px" }}
      />
      <br />
      <button 
        onClick={predict} 
        disabled={loading} 
        aria-label="Predict if the news is real or fake"
        style={{ padding: "10px 20px", backgroundColor: "#4CAF50", color: "white", border: "none", borderRadius: "5px", cursor: "pointer" }}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {/* Error Message */}
      {error && (
        <div 
          style={{ 
            color: "red", 
            padding: "10px", 
            border: "1px solid red", 
            borderRadius: "5px", 
            marginTop: "10px" 
          }}
        >
          Error: {error}
        </div>
      )}

      {/* Prediction Result */}
      {prediction && <h2>{prediction}</h2>}

      {/* Loading Indicator */}
      {loading && <div className="spinner"></div>}

      {/* Clear Button */}
      <button onClick={() => { setTextInput(''); setPrediction(''); setError(null); }} 
        style={{ padding: "10px 20px", backgroundColor: "#FF6347", color: "white", border: "none", borderRadius: "5px", cursor: "pointer", marginTop: "10px" }}>
        Clear
      </button>

    </div>
  );
}

export default App;

