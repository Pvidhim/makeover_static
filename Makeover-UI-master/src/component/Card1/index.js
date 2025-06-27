// import React, { useState, useRef } from "react";
// import "./main.css";
// import Compare from "react-compare-image";
// import ModuleBtn from "./ModuleBtn";
// import OpeningText from "./OpeningText";

// const Card = () => {
//   const fileInputRef = useRef(null);

//   // State to hold the uploaded image in base64 format
//   const [base64Image, setBase64Image] = useState("");

//   // State to hold the processed image received from the server
//   const [processedImage, setProcessedImage] = useState("");

//   const handleFileInputChange = (event) => {
//     const file = event.target.files[0];
//     const reader = new FileReader();
//     reader.onloadend = () => {
//       setBase64Image(reader.result);
//       sendImage(reader.result); // Send image after it's read
//     };
//     reader.readAsDataURL(file);
//   };

//   const sendImage = (base64Image) => {
//     const data = { image: base64Image };
//     console.log("Sending image to server:", data); // Debug log
//     fetch("http://192.168.1.16:4999/lips", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify(data),
//     })
//       .then((response) => response.json())
//       .then((data) => {
//         console.log("Response:", data);
//         if (data.image) {
//           setProcessedImage(`data:image/png;base64,${data.image}`);
//         }
//       })
//       .catch((error) => {
//         console.error("Error:", error);
//       });
//   };

//   // const card = () => {
//   return (
//     <div>
//       <div className="container">
//         {/* <!-- <div className="header"> --> */}
//         <nav className="navbar">
//           <a href="#">Home</a>
//           <a href="#">About</a>
//           <a href="#">Products</a>
//           {/* <a href="#">Buy Now</a> */}
//         </nav>
//         {/* <!-- </div> --> */}
//         <div className="background"></div>
//         <section className="section">
//           <h2 className="black-header">Try</h2>
//           {/* <p className="featured-title">Featured Product</p> */}
//           <h1 className="title">Virtual Try-On</h1>
//           {/* <h2 className="subtitle">
//             Create a <span className="highlight">shine</span>
//           </h2>
//           <h2>
//             that <span className="highlight">lasts</span>
//           </h2> */}
//           <img
//             className="product"
//             src="https://www.maccosmetics.com/media/export/cms/products/640x600/mac_sku_S3HT1M_640x600_0.jpg"
//             alt="lipglass"
//           />
//           {/* <!-- <div className="button" id="magical" onclick="magicalColor()"></div> 
//                 <div className="button" id="luxe" onclick="luxeColor()"></div>
//                 <div className="button" id="oyster" onclick="oysterColor()"></div>
//                 <div className="button" id="ruby" onclick="rubyColor()"></div> -->*/}
//           <p className="trending-desc">
//             Get in line for high-shine, because the lacquered lip glosses you
//             knew and loved from the 90s are BACK! Despite the current
//             superslick, wet and juicy lip gloss trend, our iconic Lipglass has
//             ALWAYS been poppin thanks to its moisturizing glass-like finish
//             that wears for hours.
//           </p>
//           <h1 className="tagline">Bring your vision to life</h1>
//           {/* <!-- <h1 className="tagline" id="lipglass-color">Muted Rose Pink</h1> --> */}

//           <div className="image-carousel flex">
//             <img
//               className="miniImage"
//               id="first"
//               src="https://www.maccosmetics.com/media/export/cms/products/640x600/mac_sku_S3HT1M_640x600_0.jpg"
//               onclick="swapImage(this.id)"
//             />
//             <img
//               className="miniImage"
//               id="second"
//               src="https://www.maccosmetics.com/media/export/cms/products/640x600/mac_sku_S3HT1M_640x600_1.jpg"
//               onclick="swapImage(this.id)"
//             />
//             <img
//               className="miniImage"
//               id="third"
//               src="https://www.maccosmetics.com/media/export/cms/products/640x600/mac_sku_S3HT1M_640x600_2.jpg"
//               onclick="swapImage(this.id)"
//             />
//             <img
//               className="miniImage"
//               id="fourth"
//               src="https://www.maccosmetics.com/media/export/cms/products/640x600/mac_sku_S3HT1M_640x600_3.jpg"
//               onclick="swapImage(this.id)"
//             />
//             <img
//               className="miniImage"
//               id="fifth"
//               src="https://www.maccosmetics.com/media/export/cms/products/640x600/mac_sku_S3HT1M_640x600_4.jpg"
//               onclick="swapImage(this.id)"
//             />
//           </div>
//           <div className="flex">
//             <div className="button" id="magical" onclick="magicalColor()"></div>
//             <div className="button" id="luxe" onclick="luxeColor()"></div>
//             <div className="button" id="oyster" onclick="oysterColor()"></div>
//             <div className="button" id="myth" onclick="mythColor()"></div>
//             <div className="button" id="candy" onclick="candyColor()"></div>
//             <div className="button" id="ruby" onclick="rubyColor()"></div>

//             <div className="button" id="magical" onclick="magicalColor()"></div>
//             <div className="button" id="luxe" onclick="luxeColor()"></div>
//             <div className="button" id="oyster" onclick="oysterColor()"></div>
//             <div className="button" id="myth" onclick="mythColor()"></div>
//             <div className="button" id="candy" onclick="candyColor()"></div>
//             <div className="button" id="ruby" onclick="rubyColor()"></div>
//           </div>
//           <div className="flex">
//             <div className="button" id="magical" onclick="magicalColor()"></div>
//             <div className="button" id="luxe" onclick="luxeColor()"></div>
//             <div className="button" id="oyster" onclick="oysterColor()"></div>
//             <div className="button" id="myth" onclick="mythColor()"></div>
//             <div className="button" id="candy" onclick="candyColor()"></div>
//             <div className="button" id="ruby" onclick="rubyColor()"></div>

//             <div className="button" id="magical" onclick="magicalColor()"></div>
//             <div className="button" id="luxe" onclick="luxeColor()"></div>
//             <div className="button" id="oyster" onclick="oysterColor()"></div>
//             <div className="button" id="myth" onclick="mythColor()"></div>
//             <div className="button" id="candy" onclick="candyColor()"></div>
//             <div className="button" id="ruby" onclick="rubyColor()"></div>
//           </div>
//           <p className="desc">Copyright © Visuareal 2025</p>
//         </section>
//       </div>
//     </div>
//   );
// };

// export default Card;
