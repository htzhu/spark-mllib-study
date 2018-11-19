package com.htzhu.als

// 1::Toy Story (1995)::Adventure|Animation|Children|Comedy|Fantasy
// MovieID::Title::Genres
case class Movie(mid: Int, title: String, genres: String)

// 1::122::5::838985046
// UserID::MovieID::Rating::Timestamp
case class MovieRating(uid: Int, mid: Int, rating: Double, timestamp: Long)

// 15::4973::excellent!::1215184630
// UserID::MovieID::Tag::Timestamp
case class tag(uid: Int, mid: Int, tag: String, timestamp: Long)
