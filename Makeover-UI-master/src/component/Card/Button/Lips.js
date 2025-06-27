import React, { useState, useEffect } from "react";
import { getShadesAndEndpoint } from "./sharedutils";

const ProcessLips = ({ base64Image, onProcessComplete, showShades }) => {
  const [showShadesState, setShowShadesState] = useState(showShades || false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [matteSubCategory, setMatteSubCategory] = useState("color_buzz_lip_crayon");

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
        ? { image: base64Image, liner_shade: shade, type }
        : { image: base64Image, shade, type };
  
    fetch("http://localhost:4999/lips", {
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
          // Keep shades and selection visible
          setShowShadesState(true);
          // Do NOT reset selectedCategory!
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
      {!isProcessing && (
        <div className="mt-4 text-center">
          {/* Always show category buttons */}
          <div className="flex justify-center gap-4 mb-6">
            <button
              className={`py-2 px-4 rounded-lg text-white ${selectedCategory === "matte" ? "bg-pink-700" : "bg-pink-500"}`}
              onClick={() => setSelectedCategory("matte")}
            >
              Matte
            </button>
            <button
              className={`py-2 px-4 rounded-lg text-white ${selectedCategory === "glossy" ? "bg-pink-600" : "bg-pink-400"}`}
              onClick={() => setSelectedCategory("glossy")}
            >
              Glossy
            </button>
            <button
              className={`py-2 px-4 rounded-lg text-white ${selectedCategory === "liner" ? "bg-red-500" : "bg-red-300"}`}
              onClick={() => setSelectedCategory("liner")}
            >
              Lipliner
            </button>
          </div>

          {/* Matte Shades */}
          {selectedCategory === "matte" && (
           <>
            {/* Subcategories */}
            <h2 className="text-lg font-semibold mb-3">Choose Matte Collection</h2>
            <div className="flex justify-center gap-3 mb-4 flex-wrap">
              {[
                { label: "Color Buzz Lip Crayon", value: "color_buzz_lip_crayon" },
                { label: "Color Lux", value: "color_lux" },
                { label: "Color Max", value: "color_max" },
                { label: "Kiss Sensation", value: "kiss_sensation" },
              ].map(({ label, value }) => (
                <button
                  key={value}
                  onClick={() => setMatteSubCategory(value)}
                  className={`px-4 py-2 rounded-full border ${
                    matteSubCategory === value
                      ? "bg-pink-700 text-white"
                      : "bg-white text-black hover:bg-gray-200"
                  } transition`}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Matte Shades by Subcategory */}
            <h2 className="text-lg font-semibold mb-2">
              Select {matteSubCategory.replace(/_/g, " ")} Shades
            </h2>
            <div className="flex flex-wrap justify-center gap-2">
            {matteShades
              .filter((shade) => shade.subcategory === matteSubCategory)
              .map((shade, index) => (
                <div key={index} className="flex flex-col items-center w-14 mx-1">
                  <button
                    onClick={() => sendImage(base64Image, shade.color, "matte")}
                    className="w-10 h-10 rounded-full border-2 border-white shadow"
                    style={{ backgroundColor: shade.color }}
                  />
                  <span className="mt-1 text-[10px] text-center truncate w-full">{shade.name}</span>
                </div>
              ))}


            </div>
          </>
        )}


          {/* Glossy Shades */}
          {selectedCategory === "glossy" && (
            <>
              <h2 className="text-lg font-semibold mb-2">Select Glossy Shades</h2>
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
            </>
          )}

          {/* Lipliner Shades */}
          {selectedCategory === "liner" && (
            <>
              <h2 className="text-lg font-semibold mb-2">Select Lipliner Shades</h2>
              <div className="flex flex-wrap justify-center gap-2">
                {lipLinerShades.map((shade, index) => (
                  <button
                    key={index}
                    onClick={() => sendImage(base64Image, shade.color, "liner")}
                    className="py-3 px-6 rounded-xl text-black"
                    style={{ backgroundColor: shade.color }}
                  ></button>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {isProcessing && (
        <div className="text-blue-500 py-2 px-4">Processing...</div>
      )}
    </div>
  );
};

export default ProcessLips;
