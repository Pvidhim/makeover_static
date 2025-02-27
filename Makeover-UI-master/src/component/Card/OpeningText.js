import React from "react";

const OpeningText = () => {
  return (
    <>
      <h3 className="text-3xl pb-6">
        Welcome to <br />
        <span className="font-bold text-pink-600">Virtual Try-On</span>
      </h3>
      <div className="drop-shadow-md bg-white border flex flex-col items-center pt-6">
        <h3 className="font-bold">Photo Guidelines</h3>
        <p className="max-w-md px-6 py-1">
          When choosing the ideal photo for your Virtual Try-On,
          <br /> follow these guidelines.
        </p>
        <ol className="list-decimal mx-10 pl-3 pt-10 pb-6 flex flex-col items-start">
          <li>Use a photo that is of the face straight on.</li>
          <li>Make sure nothing is obstructing the face.</li>
          <li>Make sure that the lighting is not too dim or too bright.</li>
        </ol>
      </div>
    </>
  );
};

export default OpeningText;
