// import React, { useState } from "react";
// import { getShadesAndEndpoint } from "./sharedutils";

// const ProcessEyeliner = ({ base64Image, onProcessComplete, onLipShadeReset }) => {
//   const [originalImage] = useState(base64Image); // Store the original image without eyeliner
//   const { shades, endpoint } = getShadesAndEndpoint(false); // Eyeliner is false

//   const sendImage = (base64Image, shade = null) => {
//     const data = { image: base64Image };
//     if (shade) data.shade = shade;

//     fetch(`${process.env.REACT_APP_BACKEND_URL || "http://192.168.1.9:4999"}/${endpoint}`, {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify(data),
//     })
//       .then((response) => response.json())
//       .then((data) => {
//         if (data.image) {
//           onProcessComplete(`data:image/png;base64,${data.image}`);
//         }
//       })
//       .catch((error) => {
//         console.error("Error:", error);
//       });
//   };

//   const handleShadeClick = (shade) => {
//     sendImage(originalImage, shade.color); // Apply eyeliner to the original image
//   };

//   // Reset lipstick shades when this component is active
//   if (onLipShadeReset) {
//     onLipShadeReset();
//   }

//   return (
//     <div>
//       <p className="text-md py-2">Select an Eyeliner Shade</p>
//       <div className="flex flex-wrap justify-center space-x-2">
//         {shades.map((shade, index) => (
//           <button
//             key={index}
//             onClick={() => handleShadeClick(shade)}
//             className="py-3 px-6 mt-3 rounded-xl text-white"
//             style={{ backgroundColor: shade.color }}
//           ></button>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default ProcessEyeliner;


import React, { useState, useEffect } from "react";
import { getShadesAndEndpoint } from "./sharedutils";

const ProcessEyeliner = ({ base64Image, onProcessComplete, onLipShadeReset }) => {
  const [originalImage] = useState(base64Image); // Store the original image without eyeliner
  const { shades, endpoint } = getShadesAndEndpoint(false); // Eyeliner is false
  const [processedImage, setProcessedImage] = useState(null);

  const sendImage = (base64Image, shade = null) => {
    const data = { image: base64Image };
    
    if (shade) {
      data.color = shade.color;
    }
    
    data.thickness = 6;  // Default thickness if needed
    data.transparency = 0.1;  // Default transparency if needed
    data.glow_intensity = 1;  // Default glow intensity if needed
  
    console.log("Data being sent to the backend:", data);
  
    fetch(`http://192.168.1.24:4999/${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.image) {
          setProcessedImage(`data:image/png;base64,${data.image}`);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  const handleShadeClick = (shade) => {
    sendImage(originalImage, shade); // Pass shade to apply the eyeliner color
  };

  useEffect(() => {
    // Trigger process completion once processedImage is updated
    if (processedImage) {
      onProcessComplete(processedImage);
    }
  }, [processedImage, onProcessComplete]);

  useEffect(() => {
    // Reset lipstick shades when this component is active
    if (onLipShadeReset) {
      onLipShadeReset();
    }
  }, [onLipShadeReset]);

  return (
    <div>
      <p className="text-md py-2">Select an Eyeliner Shade</p>
      <div className="flex flex-wrap justify-center space-x-2">
        {shades.map((shade, index) => (
          <button
            key={index}
            onClick={() => handleShadeClick(shade)}
            className="py-3 px-6 mt-3 rounded-xl text-white"
            style={{ backgroundColor: shade.color }}
          >
            {/* Button with no label */}
          </button>
        ))}
      </div>
    </div>
  );
};

export default ProcessEyeliner;



