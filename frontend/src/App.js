import React, { useState } from "react";
import Search from "./Search";
import Results from "./Results";
import "./index.css"; // Import styles

function App() {
  const [results, setResults] = useState([]);

  return (
    <div className="app-container">
      <h1>AI Semantic Search</h1>
      <Search setResults={setResults} />
      <Results results={results} />
    </div>
  );
}

export default App;
