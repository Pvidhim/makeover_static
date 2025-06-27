import React, { useState, useRef } from "react";
import Compare from "react-compare-image";
import OpeningText from "./OpeningText";
import ProcessLips from "./Button/Lips";
import ProcessEyeshadow from "./Button/eyeshadow";
import ProcessEyeliner from "./Button/eyeliner";
import Footer from "../Footer";
import ProcessConcealer from "./Button/concealer";
import ProcessEyebrows from "./Button/eyebrow";
import ProcessBlush from "./Button/blush";
import ProcessHair from "./Button/hair";
import ProcessIris from "./Button/iris";
const Card = () => {
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [base64Image, setBase64Image] = useState("");
  const [processedImage, setProcessedImage] = useState("");
  const [isImageProcessed, setIsImageProcessed] = useState(false);
  const [isImageUploaded, setIsImageUploaded] = useState(false);
  const [activeProcess, setActiveProcess] = useState(null); // Set default as null to hide shades initially

  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedFile(file); // Store file for uploading

    const reader = new FileReader();
    reader.onloadend = () => {
      setBase64Image(reader.result);
      setIsImageProcessed(false);
      setIsImageUploaded(false);
    };
    reader.readAsDataURL(file);
  };

  const handleUploadImage = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("http://localhost:4999/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.imageUrl) {
        setIsImageUploaded(true);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
    }
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
                    Select Image
                  </button>
                </div>
              </div>
            )}

            {base64Image && !isImageUploaded && (
              <div className="mt-3">
                <h3 className="text-3xl font-bold pb-6">Preview Image</h3>
                <img src={base64Image} width={300} alt="Selected" />
                <div className="mt-6">
                  <button
                    onClick={handleUploadImage}
                    className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
                  >
                    Upload Image
                  </button>
                </div>
              </div>
            )}

            {base64Image && isImageUploaded && !isImageProcessed && (
              <div className="mt-3">
                <h3 className="text-3xl font-bold pb-6">Virtual Try-On</h3>
                <div>
                  <img src={base64Image} width={300} alt="Uploaded" />
                </div>
                <div className="border-x border-t rounded-tl-xl rounded-tr-xl mt-6">
                  <div className="max-w-80 pb-10">
                    <h3 className="text-xl py-6">Try Products</h3>
                    <div className="flex flex-col items-center">
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
                      
                      <button
                        onClick={() => setActiveProcess("Eyeliner")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Eyeliner"
                            ? "bg-teal-500 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Eyeliner
                      </button>
                      </div>
                      <div className="flex justify-center gap-4 mt-2">
                      <button
                        onClick={() => setActiveProcess("Eyebrows")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Eyebrows"
                            ? "bg-yellow-900 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Eyebrows
                      </button>
                      <button
                        onClick={() => setActiveProcess("Concealer")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Concealer"
                            ? "bg-orange-600 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Concealer
                      </button>
                      <button
                        onClick={() => setActiveProcess("Blush")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Blush"
                            ? "bg-pink-500 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Blush
                      </button>
                      <button
                        onClick={() => setActiveProcess("Hair")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Hair"
                            ? "bg-lime-700 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Hair
                      </button>
                      <button
                        onClick={() => setActiveProcess("Iris")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Iris"
                            ? "bg-indigo-600 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                         Iris
                      </button>
                    </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="flex justify-center space-x-4">
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
                          showShades={true}  // No shades for eyeshadow
                        />
                      )}
                      {activeProcess === "Eyeliner" && (
                        <ProcessEyeliner
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  // Only show eyeliner shades when "eyeliner" is selected
                        />
                      )}
                      </div>
                      <div className="flex justify-center gap-4 mt-2">
                      {activeProcess === "Eyebrows" && (
                        <ProcessEyebrows
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  
                        />
                      )}
                      {activeProcess === "Concealer" && (
                        <ProcessConcealer
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  
                        />
                      )}
                      {activeProcess === "Blush" && (
                        <ProcessBlush
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}
                        />
                      )}
                      {activeProcess === "Hair" && (
                        <ProcessHair
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}
                        />
                      )}
                      {activeProcess === "Iris" && (
                        <ProcessIris
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}
                        />
                      )}
                      </div>
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
                    <div className="flex flex-col items-center">
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
                      <button
                        onClick={() => setActiveProcess("Eyeliner")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Eyeliner"
                            ? "bg-teal-500 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Eyeliner
                      </button>
                      </div>
                      <div className="flex justify-center gap-4 mt-2">
                      <button
                        onClick={() => setActiveProcess("Eyebrows")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Eyebrows"
                            ? "bg-yellow-900 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Eyebrows
                      </button>
                      <button
                        onClick={() => setActiveProcess("Concealer")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Concealer"
                            ? "bg-orange-600 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Concealer
                      </button>
                      <button
                        onClick={() => setActiveProcess("Blush")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Blush"
                            ? "bg-pink-600 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Blush
                      </button>
                      <button
                        onClick={() => setActiveProcess("Hair")}
                        className={`py-2 px-4 rounded ${
                          activeProcess === "Hair"
                            ? "bg-lime-700 text-white"
                            : "bg-gray-200 text-black"
                        }`}
                      >
                        Hair
                      </button>
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                    <div className="flex flex-col items-center">
                      <div className="flex justify-center space-x-4">
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
                          showShades={true}  // No shades for eyeshadow
                        />
                      )}
                      {activeProcess === "Eyeliner" && (
                        <ProcessEyeliner
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  
                        />
                      )}
                      </div>
                      <div className="flex justify-center gap-4 mt-2">
                      {activeProcess === "Eyebrows" && (
                        <ProcessEyebrows
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  
                        />
                      )}
                      {activeProcess === "Concealer" && (
                        <ProcessConcealer
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}  
                        />
                      )}
                      {activeProcess === "Blush" && (
                        <ProcessBlush
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}
                        />
                      )}
                      {activeProcess === "Hair" && (
                        <ProcessHair
                          base64Image={base64Image}
                          onProcessComplete={handleProcessedImage}
                          showShades={true}
                        />
                      )}

                      </div>
                      </div>
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