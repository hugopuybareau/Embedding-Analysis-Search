import React, { useState } from "react";
import axios from "axios";

const Search = ({ setResults }) => {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    if (!query.trim()) {
      setError("Please enter a search query.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/search/", {
        query: query.trim(),
        how_much_results: 3, // Default number of results
      });
      setResults(response.data.results);
    } catch (error) {
      setError("Error fetching results. Check API connection.");
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="search-container">
      <input
        type="text"
        placeholder="Type your query..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="search-input"
      />
      <button onClick={handleSearch} className="search-button">
        {loading ? "Searching..." : "Search"}
      </button>
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default Search;
