import React from "react";

const Results = ({ results }) => {
  return (
    <div className="results-container">
      <h2>Search Results</h2>
      {results.length === 0 ? (
        <p>No results found.</p>
      ) : (
        results.map((item, index) => (
          <div key={index} className="result-item">
            <strong>{item.document}</strong>
            <p>Relevance Score: {item.score}</p>
          </div>
        ))
      )}
    </div>
  );
};

export default Results;
