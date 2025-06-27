import React, { useState, useEffect } from "react";
import { getShadesAndEndpoint } from "./sharedutils";

const ProcessEyebrows = ({ base64Image, onProcessComplete, showShades }) => {
  const [showShadesState, setShowShadesState] = useState(showShades || false);
  const [isProcessing, setIsProcessing] = useState(false);

  const { shades, endpoint } = getShadesAndEndpoint("eyebrows"); // Fetching eyebrow shades and endpoint

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

    const data = { image: imageToSend };
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
          onClick={() => setShowShadesState(true)}
          className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
        >
          Eyebrows
        </button>
      )}
      {isProcessing && (
        <div className="text-blue-500 py-2 px-4">Processing...</div>
      )}
      {showShadesState && !isProcessing && (
        <div className="mt-4">
          <p className="text-md py-2">Select an Eyebrow Shade</p>
          <div className="flex flex-wrap justify-center gap-2 mt-4">
            {shades.map((shade, index) => (
              <button
                key={index}
                onClick={() => sendImage(base64Image, shade)}
                className="py-3 px-6 mt-3 rounded-xl text-white transition-shadow"
                style={{ backgroundColor: shade.color }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessEyebrows;
