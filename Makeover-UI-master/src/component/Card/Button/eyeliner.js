import React, { useState, useEffect } from "react";
import { getShadesAndEndpoint } from "./sharedutils";

const eyelinerStyles = ["wing", "thin", "thin+wing", "normal"];

const ProcessEyeliner = ({ base64Image, onProcessComplete, showShades }) => {
  const [showShadesState, setShowShadesState] = useState(showShades || false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState("wing"); // ✅ Default style
  const { shades, endpoint } = getShadesAndEndpoint("eyeliner");

  useEffect(() => {
    if (showShades) {
      setShowShadesState(true);
    }
  }, [showShades]);

  const sendImage = (base64Image, shade = null) => {
    setIsProcessing(true);

    let imageToSend = base64Image.startsWith("data:image")
      ? base64Image
      : `data:image/png;base64,${base64Image}`;

    const data = {
      image: imageToSend,
      style: selectedStyle, // ✅ Include selected style
    };
    if (shade) data.shade = shade.color;

    console.log("Sending data:", JSON.stringify(data));
    console.log("Using API Endpoint:", endpoint);

    fetch(`http://localhost:4999/${endpoint}`, {
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
          onClick={() => {
            setShowShadesState(true);
          }}
          className="bg-teal-500 text-white py-2 px-4 rounded hover:bg-teal-600"
        >
          Eyeliner
        </button>
      )}

      {isProcessing && (
        <div className="text-blue-500 py-2 px-4">Processing...</div>
      )}

      {showShadesState && !isProcessing && (
        <div className="mt-4 space-y-4">
          <div>
            <p className="text-md mb-2">Select Eyeliner Style</p>
            <select
              value={selectedStyle}
              onChange={(e) => setSelectedStyle(e.target.value)}
              className="border border-gray-300 rounded px-4 py-2"
            >
              {eyelinerStyles.map((style) => (
                <option key={style} value={style}>
                  {style.charAt(0).toUpperCase() + style.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div>
            <p className="text-md py-2">Select an Eyeliner Shade</p>
            <div className="flex flex-wrap justify-center space-x-2">
              {shades.map((shade, index) => (
                <button
                  key={index}
                  onClick={() => sendImage(base64Image, shade)}
                  className="py-3 px-6 mt-3 rounded-xl text-white"
                  style={{ backgroundColor: shade.color }}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessEyeliner;
