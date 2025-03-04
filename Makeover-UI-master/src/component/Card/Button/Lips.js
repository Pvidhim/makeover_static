import React, { useState, useEffect } from "react";
import { getShadesAndEndpoint } from "./sharedutils";

const ProcessLips = ({ base64Image, onProcessComplete, showShades }) => {
  const [showShadesState, setShowShadesState] = useState(showShades || false);
  const [isProcessing, setIsProcessing] = useState(false);

  // Fetch categorized lipstick and liner shades
  const { matteShades, glossyShades } = getShadesAndEndpoint("lipstick");
  const { shades: lipLinerShades } = getShadesAndEndpoint("lip_liner");

  useEffect(() => {
    if (showShades) {
      setShowShadesState(true);
    }
  }, [showShades]);

  const sendImage = (base64Image, shade, type) => {
    setIsProcessing(true);
    const data =
      type === "liner"
        ? { image: base64Image, liner_shade: shade, type } // Send liner shade separately
        : { image: base64Image, shade, type }; // Normal lipstick shades

    fetch("http://192.168.1.21:4999/lips", {
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
      {showShadesState && !isProcessing && (
        <div className="mt-4 text-center">
          {/* Matte Lipstick Section */}
          <h2 className="text-lg font-semibold mb-2">Select Matte Shades</h2>
          <div className="flex flex-wrap justify-center gap-2">
            {matteShades.map((shade, index) => (
              <button
                key={index}
                onClick={() => sendImage(base64Image, shade.color, "matte")}
                className="py-3 px-6 rounded-xl text-white"
                style={{ backgroundColor: shade.color }}
              ></button>
            ))}
          </div>

          {/* Glossy Lipstick Section */}
          <h2 className="text-lg font-semibold mt-6 mb-2">Select Glossy Shades</h2>
          <div className="flex flex-wrap justify-center gap-2">
            {glossyShades.map((shade, index) => (
              <button
                key={index}
                onClick={() => sendImage(base64Image, shade.color, "glossy")}
                className="py-3 px-6 rounded-xl relative"
                style={{
                  backgroundColor: shade.color,
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                {/* Glossy Effect Overlay */}
                <div
                  className="absolute top-0 left-0 w-full h-full"
                  style={{
                    background: `linear-gradient(135deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0.1) 50%, transparent 100%)`,
                    opacity: 0.6,
                    borderRadius: "12px",
                    pointerEvents: "none",
                  }}
                />
              </button>
            ))}
          </div>

          {/* Lip Liner Section */}
          <h2 className="text-lg font-semibold mt-6 mb-2">Select Lip Liner</h2>
          <div className="flex flex-wrap justify-center gap-2">
            {lipLinerShades.map((shade, index) => (
              <button
                key={index}
                onClick={() => sendImage(base64Image, shade.color, "liner")} // Use "liner" type
                className="py-3 px-6 rounded-xl text-black"
                style={{ backgroundColor: shade.color }}
              ></button>
            ))}
          </div>
        </div>
      )}

      {isProcessing && <div className="text-blue-500 py-2 px-4">Processing...</div>}
    </div>
  );
};

export default ProcessLips;
