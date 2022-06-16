/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ailab.aicardio.repository

import android.app.Activity
import android.content.Context
import java.util.logging.Logger

data class User(
    val name: String = "",
    val phone: String = ""
) {
    companion object {
        const val PREFS_NAME = "ACCOUNT"
        const val KEY_PHONE = "KEY_PHONE"
        const val KEY_PASSWORD = "KEY_PASSWORD"
        const val KEY_FULL_NAME = "KEY_FULL_NAME"
        const val DEFAULT_PHONE = "LOGIN"

        fun getPhone(context: Context) : String? {
            val mPref = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return mPref.getString(KEY_PHONE, DEFAULT_PHONE)
        }

        fun remember(activity: Activity, phone: String){
//            Logger.getLogger(User.javaClass.name).warning("remember: $phone, $password")
            val editor = activity.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            editor.putString(KEY_PHONE, phone)
//            editor.putString(KEY_PASSWORD, password)
//            editor.putString(KEY_FULL_NAME, fullName)
            editor.apply()
        }



    }
}