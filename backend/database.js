import { Pool } from "pg";
import dotenv from "dotenv";
import { errorThrower } from "./utils/errorthrower.js";

dotenv.config();

export const pool = new Pool({
  connectionString: process.env.SUPABASE_CONNECTION_STRING,
  ssl: { rejectUnauthorized: false },
});

export const checkConnection = async () => {
  try {
    const result = await pool.query("SELECT NOW()");
    console.log(
      `Database connected at time: ${JSON.stringify(result.rows[0].now)}`
    );
  } catch (err) {
    console.log("Error connecting to database: ", err.message);
  }
};

export const addUserDB = async (username, password) => {
  try {
    const result = await pool.query(
      "INSERT INTO users (username, password) VALUES ($1, $2) RETURNING *",
      [username, password]
    );
    return result;
  } catch (error) {
    throw errorThrower("Failed to create user, ", error.message);
  }
};

export const getUserByUsernameDB = async (username) => {
  try {
    const result = await pool.query("SELECT * FROM USERS WHERE username=$1", [
      username,
    ]);
    return result;
  } catch (error) {
    throw errorThrower("Failed to get user details: ", error.nessage);
  }
};

export const addPredictionDB = async (userid, imageUrl, disease, reason) => {
  try {
    await pool.query(
      "INSERT INTO detections (imageurl, userid, disease, reason) VALUES ($1, $2, $3, $4)",
      [imageUrl, userid, disease, reason]
    );
  } catch (error) {
    throw errorThrower("Failed to add detections. ", error.message);
  }
};

// for a particular user to be displayed in user profile.
export const getAllUserPredictionsDB = async (userid) => {
  try {
    const result = await pool.query(
      "SELECT * FROM detections WHERE userid=$1 ORDER BY created_at DESC",
      [userid]
    );
    return result.rows; // array of prediction objects
  } catch (error) {
    throw errorThrower("Failed to get past user predictions. ", error.message);
  }
};