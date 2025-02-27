import React, { useState, useRef } from "react";
import Compare from "react-compare-image";
import OpeningText from "./OpeningText";
import ProcessLips from "./Button/Lips";
import ProcessEyeshadow from "./Button/eyeshadow";
import Footer from "../Footer";

const Card = () => {
  const fileInputRef = useRef(null);

  const [base64Image, setBase64Image] = useState("");
  const [processedImage, setProcessedImage] = useState("");
  const [isImageProcessed, setIsImageProcessed] = useState(false);
  const [activeProcess, setActiveProcess] = useState(null);  // Set default as null to hide shades initially

  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onloadend = () => {
      setBase64Image(reader.result);
      setIsImageProcessed(false);
    };
    reader.readAsDataURL(file);
  };

  const handleProcessedImage = (processedImage) => {
    setProcessedImage(processedImage);
    setIsImageProcessed(true);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white pt-6 px-6 rounded-lg shadow-lg text-center">
        <div className="flex flex-col justify-center px-3">
          <div className="px-3 rounded-lg">
            {!base64Image && (
              <div className="flex flex-col items-center justify-center">
                <OpeningText />
                <div className="flex flex-col items-center justify-center pt-6 pb-10">
                  <input
                    type="file"
                    onChange={handleFileInputChange}
                    ref={fileInputRef}
                    style={{ display: "none" }}
                  />
                  <button
                    onClick={() => fileInputRef.current.click()}
                    className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
                  >
                    Upload Image
                  </button>
                </div>
              </div>
            )}
            {base64Image && !isImageProcessed && (
              <div className="mt-3">
                <h3 className="text-3xl font-bold pb-6">Virtual Try-On</h3>
                <div>
                  <img src={base64Image} width={300} alt="Uploaded" />
                </div>
                <div className="border-x border-t rounded-tl-xl rounded-tr-xl mt-6">
                  <div className="max-w-80 pb-10">
                    <h3 className="text-xl py-6">Try Products</h3>
                    <div className="flex justify-center space-x-4">
                      <button
                        onClick={() => setActiveProcess("Lips")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Lips"
                            ? "bg-blue-600 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Lipstick
                      </button>
                      <button
                        onClick={() => setActiveProcess("Eyeshadow")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Eyeshadow"
                            ? "bg-purple-500 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Eyeshadow
                      </button>
                    </div>
                    <div className="flex justify-center mt-6">
                      {activeProcess === "Lips" && (
                        <ProcessLips
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  // Only show lipstick shades when "Lipstick" is selected
                        />
                      )}
                      {activeProcess === "Eyeshadow" && (
                        <ProcessEyeshadow
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={false}  // No shades for eyeshadow
                        />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
            {isImageProcessed && (
              <div className="mt-3">
                <h3 className="text-3xl font-bold pb-6">Virtual Try-On</h3>
                <Compare
                  leftImage={base64Image}
                  rightImage={processedImage}
                  sliderLineColor="white"
                  sliderPositionPercentage={0.5}
                  sliderLineWidth={1}
                />
                <div className="border-x border-t border-inherit rounded-tl-xl rounded-tr-xl mt-6">
                  <div className="max-w-80 pb-10">
                    <h3 className="text-xl pt-6">Try Shades</h3>
                    <div className="flex justify-center space-x-4">
                      <button
                        onClick={() => setActiveProcess("Lips")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Lips"
                            ? "bg-blue-600 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Lipstick
                      </button>
                      <button
                        onClick={() => setActiveProcess("Eyeshadow")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Eyeshadow"
                            ? "bg-purple-500 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Eyeshadow
                      </button>
                    </div>
                    <div className="flex justify-center mt-6">
                      {activeProcess === "Lips" && (
                        <ProcessLips
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  // Only show lipstick shades when "Lipstick" is selected
                        />
                      )}
                      {activeProcess === "Eyeshadow" && (
                        <ProcessEyeshadow
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={false}  // No shades for eyeshadow
                        />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          <Footer />
        </div>
      </div>
    </div>
  );
};


export default Card;