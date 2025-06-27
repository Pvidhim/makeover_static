import React, { useState, useEffect } from "react";

const blushShades = [
  { name: "B_01_Soft_Peach", color: "#FFADAD" },      // Soft Peachy Pink
  { name: "B_02_Warm_Rose", color: "#FF8A8A" },       // Warm Rosy Pink
  { name: "B_03_Coral_Glow", color: "#FF7C6B" },      // Coral Tint
  { name: "B_04_Dusty_Rose", color: "#E89CA5" },      // Muted Pink Rose
  { name: "B_05_Terracotta", color: "#D98880" },      // Warm Terracotta
  { name: "B_06_Rosy_Beige", color: "#E6A4A4" },      // Natural Nude Blush
  { name: "B_07_Mauve_Pink", color: "#D291BC" },      // Mauve with a Pink Hint
  { name: "B_08_Plum_Flush", color: "#B76E79" },      // Soft Plum
];


const ProcessBlush = ({ base64Image, onProcessComplete, showShades }) => {
  const [showShadesState, setShowShadesState] = useState(showShades || false);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (showShades) {
      setShowShadesState(true);
    }
  }, [showShades]);

  const sendImage = (base64Image, shade = null) => {
    setIsProcessing(true);
    const data = { image: base64Image };
    if (shade) data.shade = shade;

    fetch("http://localhost:4999/blush", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.image) {
          onProcessComplete(`data:image/png;base64,${data.image}`);
          setShowShadesState(true);
        }
        setIsProcessing(false);
      })
      .catch((error) => {
        console.error("Error:", error);
        setIsProcessing(false);
      });
  };

  return (
    <div>
      {!showShadesState && !isProcessing && (
        <button
          onClick={() => sendImage(base64Image)}
          className="bg-pink-500 text-white py-2 px-4 rounded hover:bg-pink-600"
        >
          Blush
        </button>
      )}
      {isProcessing && (
        <div className="text-pink-500 py-2 px-4">Processing...</div>
      )}
      {showShadesState && !isProcessing && (
        <div className="mt-4">
          <p className="text-md py-2">Select a Blush Shade</p>
          <div className="flex flex-wrap justify-center space-x-2">
            {blushShades.map((shade, index) => (
              <button
                key={index}
                onClick={() => sendImage(base64Image, shade.color)}
                className="py-3 px-6 mt-3 rounded-xl text-white"
                style={{ backgroundColor: shade.color }}
              >
                {/* {shade.name} */}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessBlush;
