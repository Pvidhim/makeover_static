import React, { useState, useEffect } from "react";
import { getShadesAndEndpoint } from "./sharedutils";

const ProcessEyeshadow = ({ base64Image, onProcessComplete }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [shades, setShades] = useState([]);
  const [endpoint, setEndpoint] = useState("");

  // Fetch shades and API endpoint dynamically
  useEffect(() => {
    const { shades: fetchedShades, endpoint: fetchedEndpoint } = getShadesAndEndpoint("eyeshadow");
    
    console.log("Fetched Shades Data:", fetchedShades); // ✅ Log to verify shades data
    console.log("Fetched API Endpoint:", fetchedEndpoint);

    setShades(fetchedShades);
    setEndpoint(fetchedEndpoint);
  }, []);

  const sendImage = (base64Image, shade) => {
    if (!shade) {
      console.error("Error: Shade is undefined or empty!"); // ✅ Debugging log
      return;
    }

    setIsProcessing(true);
    console.log("Selected Shade:", shade); // ✅ Log selected shade before sending

    const data = { image: base64Image, shade };

    fetch(`http://192.168.1.21:4999/${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Response from Backend:", data); // ✅ Log backend response

        if (data.image) {
          onProcessComplete(`data:image/png;base64,${data.image}`);
        } else {
          console.error("Error: No processed image received from backend.");
        }
        setIsProcessing(false);
      })
      .catch((error) => {
        console.error("Fetch Error:", error);
        setIsProcessing(false);
      });
  };

  return (
    <div>
      {isProcessing && <div className="text-purple-500 py-2 px-4">Processing...</div>}

      {!isProcessing && (
        <div className="mt-4">
          <p className="text-md py-2">Select an Eyeshadow Shade</p>
          <div className="flex flex-wrap justify-center space-x-2">
            {shades.length > 0 ? (
              shades.map((shade, index) => (
                <button
                  key={index}
                  onClick={() => {
                    console.log("Clicked Shade:", shade.color); // ✅ Debugging log
                    sendImage(base64Image, shade.color);
                  }}
                  className="py-3 px-6 mt-3 rounded-xl text-white"
                  style={{ backgroundColor: shade.color }}
                ></button>
              ))
            ) : (
              <p className="text-gray-500">No shades available</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessEyeshadow;


