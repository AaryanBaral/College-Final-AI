import fetch from "node-fetch";
import dotenv from "dotenv";
import FormData from "form-data";
import { errorThrower } from "../utils/errorthrower.js";
import { addPredictionDB, getAllUserPredictionsDB } from "../database.js";

dotenv.config();

export const uploadImage = async (req, res) => {
  try {
    const image = req.file;
    if (!image) {
      return res.status(400).json(errorThrower("No image provided."));
    }

    const cloudinaryData = new FormData();
    cloudinaryData.append("file", image.buffer, {
      filename: `${image.originalname}+Math.random().toString(36).slice(-6)`,
      contentType: image.mimetype,
    });
    cloudinaryData.append(
      "upload_preset",
      process.env.CLOUDINARY_UPLOAD_PRESET
    );

    const response = await fetch(
      `https://api.cloudinary.com/v1_1/${process.env.CLOUDINARY_CLOUD_NAME}/image/upload`,
      {
        method: "POST",
        body: cloudinaryData,
      }
    );
    const data = await response.json();
    if (data.secure_url) {
      return res.status(200).json({
        success: true,
        message: data.secure_url,
      });
    } else {
      return res
        .status(400)
        .json(errorThrower("Failed to upload image to Cloudinary.", data));
    }
  } catch (error) {
    return res.status(400).json(errorThrower(error.message));
  }
};

export const getPrediction = async (req, res) => {
  // logic that returns prediction and reason using model as a response
  try {
    const { imageUrl, userid } = req.body;
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-type": "application/json",
      },
      body: JSON.stringify({
        imageUrl,
      }),
    });
    const result = await response.json(); // result from predictor api
    /*
        expecting result to be of the form: 
        {
            success: true,
            message: {
                disease: "diseaseName",
                reason: "reason- not defined as of now"
            }
        }
        message will be an error message string if error occured (i.e if success: false)
    */
    if (result.success == false) {
      return res.status(400).json(errorThrower(result.message));
    }
    // save the prediction to database
    await addPredictionDB(
      userid,
      imageUrl,
      result.message.disease,
      result.message.reason
    );

    return res.status(200).json({
      success: true,
      message: result.message,
    });
  } catch (error) {
    return res.status(400).json(errorThrower(error.message));
  }
};

export const getAllUserPredictions = async (req, res) => {
  try {
    const { userid } = req.params;
    const result = await getAllUserPredictionsDB(userid);
    return res.status(200).json({
      success: true,
      message: result,
    });
  } catch (error) {
    return res
      .status(400)
      .json(errorThrower("Failed to get past predictions. ", error.message));
  }
};
