import React from 'react';
import Navbar from '../components/Navbar.jsx';
import "../styles/signup.css";
import { Link } from 'react-router';
import { useNavigate } from 'react-router';
import { useDispatch } from 'react-redux';
import { toast } from 'react-toastify';
import { signUpFailure, signUpStart, signUpSuccess } from '../redux/user/userSlice.js';

export default function Signup() {

    const navigate = useNavigate();
    const dispatch = useDispatch();
    const [isSubmitDisabled, setIsSubmitDisabled] = React.useState(false);
    const [formData, setFormData] = React.useState({
        username: null,
        password: null,
        confirmPassword: null,
    })

    const handleChange = (e)=> {
        setFormData((prevFormData) => {
            return {
                ...prevFormData,
                [e.target.id]: e.target.value,
            };
        });
    };

    const handleSubmit = async(e)=> {
        e.preventDefault();
        setIsSubmitDisabled(true);
        dispatch(signUpStart());
        try {
            const response = await fetch("http://localhost:3000/api/auth/signup", {
                method: "POST",
                headers: {
                    "Content-type": "application/json",
                },
                body: JSON.stringify(formData),
            });
            const result = await response.json();

            if (result.success == false) {
                dispatch(signUpFailure(result.message));
                toast.error(result.message);
                setIsSubmitDisabled(false);
                return;
            }
            dispatch(signUpSuccess());
            toast.success("User created successfully.")
            setIsSubmitDisabled(false);
            navigate("/signin");
            return;
        } catch(error) {
            dispatch(signUpFailure(error.message))
            toast.error(error.message);
            setIsSubmitDisabled(false);
            return;
        }
    };

    return (
        <div className='signup-page'>
                <Navbar/>
                <div className='signup-main'>
                    <form className='signup' onSubmit={handleSubmit}>
                        <div>
                            <div className='create-an-account'>Create an account</div>
                            <div className='enter-info-below'>Enter the information below to create your account</div>
                        </div>
                        <div>
                            <div className='username'>
                                <label htmlFor='username'>User Name</label>
                            </div>
                            <input id='username' type='text' className='username-input' placeholder='JohnDoe' onChange={handleChange} required/>
                        </div>
                        <div>
                            <div className='password'>
                                <label htmlFor="password">Password</label>
                            </div>
                            <input id="password" type="password" className='password-input' onChange={handleChange} required/>
                            <div className='password-constraint'>Must be at least 8 characters long.</div>
                        </div>
                        <div>
                            <div className='confirm-password'>
                                <label htmlFor="confirmPassword">Confirm Password</label>
                            </div>
                            <input id="confirmPassword" type="password" className='confirm-password-input' onChange={handleChange} required />
                            <div className='confirm-password-constraint'>Please confirm your password.</div>
                        </div>
                        <div>
                            <button className='create-account-button' type='submit' disabled={isSubmitDisabled}>{isSubmitDisabled? "Loading...": "Create Account"}</button>
                            <div className='already-have-account'>Already have an account? <Link to={"/signin"} className='route-link-signin'>Sign in</Link></div>
                        </div>
                    </form>
                </div>
            </div>
    )
}
