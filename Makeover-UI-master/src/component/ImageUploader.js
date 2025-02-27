import React, { useState } from "react";

const ImageUploader = () => {
  const [base64Image, setBase64Image] = useState("");
  const [processedImage, setProcessedImage] = useState("");

  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onloadend = () => {
      setBase64Image(reader.result);
      sendImage(reader.result);  // Send image after it's read
    };
    reader.readAsDataURL(file);
  };

  const sendImage = (base64Image) => {
    const data = { image: base64Image };
    console.log("Sending image to server:", data);  // Debug log
    fetch("http://192.168.1.24:4999/lips", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Response:", data);
        if (data.image) {
          setBase64Image(`data:image/png;base64,${data.image}`);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  return (
    <div>
      <input type="file" onChange={handleFileInputChange} />
      {base64Image && <img src={base64Image} alt="Uploaded" />}
    </div>
  );
};

export default ImageUploader;
