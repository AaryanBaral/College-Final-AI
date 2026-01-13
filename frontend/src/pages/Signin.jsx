import React from "react";
import Navbar from "../components/Navbar.jsx";
import "../styles/signin.css";
import { Link } from "react-router";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router";
import {
  signInFailure,
  signInStart,
  signInSuccess,
} from "../redux/user/userSlice.js";
import { toast } from "react-toastify";

export default function Signin() {
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const [formData, setFormData] = React.useState({
    username: null,
    password: null,
  });
  const [isSubmitDisabled, setIsSubmitDisabled] = React.useState(false);

  const handleChange = (e) => {
    setFormData((prevFormData) => {
      return {
        ...prevFormData,
        [e.target.id]: e.target.value,
      };
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitDisabled(true);
    dispatch(signInStart());
    try {
      const response = await fetch("/api/auth/signin", {
        method: "POST",
        headers: {
          "Content-type": "application/json",
        },
        body: JSON.stringify(formData),
        credentials: "include",
      });
      const result = await response.json();
      if (result.success == false) {
        dispatch(signInFailure(result.message));
        toast.error(result.message);
        setIsSubmitDisabled(false);
        return;
      }
      dispatch(signInSuccess(result.message));
      toast.success("Logged in successfully.");
      setIsSubmitDisabled(false);
      navigate("/");
      return;
    } catch (error) {
      dispatch(signInFailure(error.message));
      toast.error(error.message);
      setIsSubmitDisabled(false);
      return;
    }
  };

  return (
    <div className="signin-page">
      <Navbar />
      <div className="signin-main">
        <form className="signin" onSubmit={handleSubmit}>
          <div>
            <div className="login-to-account">Login to your account</div>
            <div className="enter-info-below">
              Enter the details below to login to your account
            </div>
          </div>
          <div>
            <div className="username">
              <label htmlFor="username">User Name</label>
            </div>
            <input
              id="username"
              type="text"
              className="username-input"
              placeholder="JohnDoe"
              onChange={handleChange}
            />
          </div>
          <div>
            <div className="password">
              <label htmlFor="password">Password</label>
            </div>
            <input
              id="password"
              type="password"
              className="password-input"
              onChange={handleChange}
            />
          </div>
          <div>
            <button
              className="login-button"
              type="submit"
              disabled={isSubmitDisabled}
            >
              {isSubmitDisabled ? "Loading..." : "Login"}
            </button>
            <div className="dont-have-account">
              Don't have an account?{" "}
              <Link to={"/signup"} className="route-link-signup">
                Sign up
              </Link>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
