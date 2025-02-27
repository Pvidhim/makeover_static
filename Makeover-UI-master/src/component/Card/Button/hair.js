import React, { useState, useEffect } from "react";

const hairShades = [
  { name: "H_01_4B3621", color: "#4B3621" },
  { name: "H_02_2C1B10", color: "#2C1B10" },
  { name: "H_03_5A3E24", color: "#5A3E24" },
  { name: "H_04_6D4A2C", color: "#6D4A2C" },
  { name: "H_05_825A3B", color: "#825A3B" },
  { name: "H_06_9B6C46", color: "#9B6C46" },
  { name: "H_07_C87A5D", color: "#C87A5D" },
  { name: "H_08_F4B97B", color: "#F4B97B" },
  { name: "H_09_7A4825", color: "#7A4825" },
  { name: "H_10_3E2A1A", color: "#3E2A1A" },
  { name: "H_11_4C3B28", color: "#4C3B28" },
  { name: "H_12_564133", color: "#564133" },
  { name: "H_13_6A4F3E", color: "#6A4F3E" },
  { name: "H_14_8C6549", color: "#8C6549" },
  { name: "H_15_7B553D", color: "#7B553D" },
  { name: "H_16_BC7E59", color: "#BC7E59" },
  { name: "H_17_8D5C37", color: "#8D5C37" },
  { name: "H_18_6F482E", color: "#6F482E" },
  { name: "H_19_5B3E28", color: "#5B3E28" },
  { name: "H_20_A4734E", color: "#A4734E" },
  { name: "H_21_966441", color: "#966441" },
  { name: "H_22_7C563D", color: "#7C563D" },
  { name: "H_23_67462F", color: "#67462F" },
  { name: "H_24_523922", color: "#523922" },
];

const ProcessHair = ({ base64Image, onProcessComplete, showShades }) => {
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
    fetch("http://192.168.1.24:4999/hair", {
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
          className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
        >
          Hair Color
        </button>
      )}
      {isProcessing && (
        <div className="text-blue-500 py-2 px-4">Processing...</div>
      )}
      {showShadesState && !isProcessing && (
        <div className="mt-4">
          <p className="text-md py-2">Select a Hair Shade</p>
          <div className="flex flex-wrap justify-center space-x-2">
            {hairShades.map((shade, index) => (
              <button
                key={index}
                onClick={() => sendImage(base64Image, shade.color)}
                className="py-3 px-6 mt-3 rounded-xl text-white"
                style={{ backgroundColor: shade.color }}
              ></button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessHair;
