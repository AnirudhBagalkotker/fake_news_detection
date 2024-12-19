import './App.css';
import React, { useState } from "react";

const App = () => {
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [isValidURL, setIsValidURL] = useState(true);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const validateURL = (url) => {
    const regex = /^(https?:\/\/)?([a-zA-Z0-9_-]+\.)+[a-zA-Z]{2,6}(\/.*)?$/;
    return regex.test(url);
  };

  const handleURLChange = (e) => {
    const value = e.target.value;
    setUrl(value);
    setIsValidURL(validateURL(value));
    if (value) {
      setTitle("");
      setText("");
    }
  };

  const handleTitleChange = (e) => {
    setTitle(e.target.value);
    if (e.target.value) setUrl("");
  };

  const handleTextChange = (e) => {
    setText(e.target.value);
    if (e.target.value) setUrl("");
  };

  const isFormValid = () => {
    if (url) return isValidURL;
    if (title && text) return true;
    return false;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isFormValid()) {
      setError("Please enter a valid URL or both title and text before submitting.");
      return;
    }

    setResult(null);
    setError(null);

    try {
      const payload = url ? { url } : { title, text };
      const endpoint = url ? "http://127.0.0.1:8000/predict" : "http://127.0.0.1:8000/prediction";
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2>Add URL or Title and Text</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>URL:</label>
          <input
            type="text"
            value={url}
            onChange={handleURLChange}
            placeholder="Enter a URL"
            style={{
              padding: "10px",
              width: "100%",
              boxSizing: "border-box",
              marginBottom: "10px",
              border: isValidURL ? "2px solid green" : "2px solid red",
              borderRadius: "5px",
            }}
          />
          {url && (
            <p style={{ color: isValidURL ? "green" : "red" }}>
              {isValidURL ? "Valid URL!" : "Invalid URL!"}
            </p>
          )}
        </div>

        <h4 style={{ textAlign: "center", margin: "10px 0" }}>OR</h4>

        <div>
          <label>Title:</label>
          <input
            type="text"
            value={title}
            onChange={handleTitleChange}
            placeholder="Enter a title"
            style={{
              padding: "10px",
              width: "100%",
              boxSizing: "border-box",
              marginBottom: "10px",
              borderRadius: "5px",
              border: title ? "2px solid green" : "2px solid #ccc",
            }}
          />
          <label>Text:</label>
          <textarea
            value={text}
            onChange={handleTextChange}
            placeholder="Enter text"
            rows="4"
            style={{
              padding: "10px",
              width: "100%",
              boxSizing: "border-box",
              marginBottom: "10px",
              borderRadius: "5px",
              border: text ? "2px solid green" : "2px solid #ccc",
            }}
          ></textarea>
        </div>

        <button
          type="submit"
          disabled={!isFormValid()}
          style={{
            padding: "10px 20px",
            backgroundColor: isFormValid() ? "#4CAF50" : "#ccc",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: isFormValid() ? "pointer" : "not-allowed",
          }}
        >
          Submit
        </button>
      </form>

      {result && (
        <div
          style={{
            marginTop: "20px",
            padding: "10px",
            border: "1px solid #ccc",
            borderRadius: "5px",
            backgroundColor: "#f9f9f9",
          }}
        >
          <h3>Result</h3>
          <p>
            <strong>Title:</strong> {result.title}
          </p>
          <p>
            <strong>Prediction:</strong>{" "}
            <span
              style={{
                color: result.prediction === "True" ? "green" : "red",
                fontWeight: "bold",
              }}
            >
              {result.prediction}
            </span>
          </p>
        </div>
      )}
      {error && (
        <div style={{ marginTop: "20px", color: "red" }}>
          <h3>Error:</h3>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default App;
