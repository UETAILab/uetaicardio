package com.ailab.aicardio.api

import com.ailab.aicardio.BASE_URL
import com.ailab.aicardio.repository.NewsApiResponse
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET

interface NewsApi {
    @GET("top-headlines?category=technology&pageSize=40&apiKey=40338dc2ef2f41ab9a302a9c750cc6f9")
    suspend fun getLatestNews(): NewsApiResponse

    companion object {
        fun create(): NewsApi {
            return Retrofit.Builder()
                .baseUrl(BASE_URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(NewsApi::class.java)
        }
    }
}