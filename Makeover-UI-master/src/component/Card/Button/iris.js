import React, { useState, useEffect } from "react";

// Available iris colors - these should match the PNG files in your iriscolors folder
const irisColors = [
  { name: "aquagreeniris", displayName: "Aqua Green", color: "#00CED1" },
  { name: "bluegreeniris", displayName: "Blue Green", color: "#0D98BA" },
  { name: "blueiris", displayName: "Blue", color: "#1E90FF" },
  { name: "bluishiris", displayName: "Bluish", color: "#4169E1" },
  { name: "darkbluesnowflake", displayName: "Dark Blue", color: "#191970" },
  { name: "greeniris", displayName: "Green", color: "#32CD32" },
  { name: "greyfloweriris", displayName: "Grey Flower", color: "#708090" },
  { name: "lightgrey", displayName: "Light Grey", color: "#D3D3D3" },
  { name: "snakeiris", displayName: "Snake", color: "#8B4513" },
  { name: "yellowiris", displayName: "Yellow", color: "#FFD700" },
];

const ProcessIris = ({ base64Image, onProcessComplete, showShades }) => {
  const [showShadesState, setShowShadesState] = useState(showShades || false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [availableColors, setAvailableColors] = useState([]);
  const [opacity, setOpacity] = useState(0.75);

  useEffect(() => {
    if (showShades) {
      setShowShadesState(true);
      fetchAvailableColors();
    }
  }, [showShades]);

  const fetchAvailableColors = async () => {
    try {
      const response = await fetch('http://localhost:4999/iris/colors');
      const data = await response.json();
      
      if (response.ok && data.colors) {
        // Map server colors with our display info
        const mappedColors = data.colors.map(colorName => {
          const colorInfo = irisColors.find(c => c.name === colorName) || 
                           { name: colorName, displayName: colorName, color: "#4169E1" };
          return colorInfo;
        });
        setAvailableColors(mappedColors);
      } else {
        // Fallback to predefined colors if server fails
        setAvailableColors(irisColors);
      }
    } catch (error) {
      console.error("Error fetching iris colors:", error);
      // Fallback to predefined colors
      setAvailableColors(irisColors);
    }
  };

  const sendImage = (base64Image, irisColorName = null, currentOpacity = opacity) => {
    setIsProcessing(true);
    const data = { 
      image: base64Image,
      iris_color: irisColorName || "bluegreeniris",
      opacity: currentOpacity
    };

    fetch("http://localhost:4999/iris", {
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
        } else if (data.error) {
          console.error("Server error:", data.error);
          alert(`Error: ${data.error}`);
        }
        setIsProcessing(false);
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Failed to apply iris color. Please try again.");
        setIsProcessing(false);
      });
  };

  const handleOpacityChange = (newOpacity) => {
    setOpacity(newOpacity);
  };

  const applyIrisWithOpacity = (irisColorName) => {
    sendImage(base64Image, irisColorName, opacity);
  };

  return (
    <div className="flex flex-col items-center">
      {showShadesState && (
        <div className="flex flex-col items-center mt-4">
          <h4 className="text-lg font-semibold mb-2"> Choose Iris Color</h4>
          
          {/* Opacity Slider */}
          <div className="mb-4 w-full max-w-xs">
            <label className="block text-sm font-medium mb-1">
              Opacity: {Math.round(opacity * 100)}%
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => handleOpacityChange(parseFloat(e.target.value))}
              disabled={isProcessing}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Iris Color Grid */}
          <div className="grid grid-cols-3 gap-3 max-w-sm">
            {availableColors.map((irisColor) => (
              <button
                key={irisColor.name}
                onClick={() => !isProcessing && applyIrisWithOpacity(irisColor.name)}
                disabled={isProcessing}
                className={`
                  flex flex-col items-center p-2 rounded-lg border-2 transition-all duration-200
                  ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-lg cursor-pointer'}
                  border-gray-300 hover:border-indigo-400
                `}
                title={irisColor.displayName}
              >
                {/* Color Preview Circle */}
                <div
                  className="w-8 h-8 rounded-full border-2 border-white shadow-sm mb-1"
                  style={{ backgroundColor: irisColor.color }}
                ></div>
                
                {/* Color Name */}
                <span className="text-xs text-center leading-tight">
                  {irisColor.displayName}
                </span>
              </button>
            ))}
          </div>

          {isProcessing && (
            <div className="mt-4 text-center">
              <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600"></div>
              <p className="text-sm text-gray-600 mt-2">Applying iris color...</p>
            </div>
          )}

          {availableColors.length === 0 && !isProcessing && (
            <div className="mt-4 text-center text-gray-500">
              <p>Loading iris colors...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ProcessIris;
