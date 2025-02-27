// import "./App.css";

// import React from "react";
// import Home from "./component/home.js";
// // import "./styles.css";

// const App = () => {
//   return (
//     <div>
//       <Home />
//     </div>
//   );
// };

// export default App;

import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./component/home.js";  // Assuming Home contains Card component
import Card from "./component/Card";    // Import the Card component

const App = () => {
  return (
    <Router>
      <Routes>
        {/* Route for the home page */}
        <Route path="/" element={<Home />} />

        {/* Route for Lipstick and Eyeliner, both using Card component */}
        <Route path="/lips" element={<Card product="lips" />} />
        
      </Routes>
    </Router>
  );
};

export default App;

