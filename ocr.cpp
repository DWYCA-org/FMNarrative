#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <map>
#include <algorithm>
#include <sstream>
#include <set>
#include <iterator>

struct MatchStatistic {
    std::string statName;
    std::string homeValue;
    std::string awayValue;
    std::string homeTeam;
    std::string awayTeam;
};

class OCRReader {
private:
    tesseract::TessBaseAPI* ocrGeneral;
    tesseract::TessBaseAPI* ocrNumbers;
    
    std::map<std::string, std::string> statMappings = {
        // REMOVED "shots" mapping - we'll calculate it
        {"on target", "on target"}, 
        {"expected goals", "xg"},
        {"xg", "xg"},
        {"xG", "xg"},
        {"off target", "off target"},
        {"clear cut chances", "clear cut chances"},
        {"clear cut chance", "clear cut chances"},
        {"long shots", "long shots"},
        {"long shot", "long shots"},
        {"possession", "possession"},
        {"corners", "corners"},
        {"corner", "corners"},
        {"fouls", "fouls"},
        {"foul", "fouls"},
        {"offsides", "offsides"},
        {"offside", "offsides"},
        {"passes completed", "passes completed"},
        {"passes complete", "passes completed"},
        {"crosses completed", "crosses completed"},
        {"crosses complete", "crosses completed"},
        {"tackles won", "tackles won"},
        {"tackle won", "tackles won"},
        {"headers won", "headers won"},
        {"header won", "headers won"},
        {"yellow cards", "yellow cards"},
        {"yellow card", "yellow cards"},
        {"yellow", "yellow cards"},
        {"red cards", "red cards"},
        {"red card", "red cards"},
        {"red", "red cards"},
        {"average rating", "average rating"},
        {"avg rating", "average rating"},
        {"rating", "average rating"},
        {"progressive passes", "progressive passes"},
        {"progressive pass", "progressive passes"},
        {"high intensity sprints", "high intensity sprints"},
        {"high intensity sprint", "high intensity sprints"},
        {"intensity sprints", "high intensity sprints"},
        {"sprints", "high intensity sprints"}
    };

public:
    OCRReader() {
        // General OCR
        ocrGeneral = new tesseract::TessBaseAPI();
        if (ocrGeneral->Init(NULL, "eng")) {
            std::cerr << "Could not initialize general tesseract." << std::endl;
            exit(1);
        }
        ocrGeneral->SetPageSegMode(tesseract::PSM_AUTO);
        
        // Numbers-only OCR
        ocrNumbers = new tesseract::TessBaseAPI();
        if (ocrNumbers->Init(NULL, "eng")) {
            std::cerr << "Could not initialize numbers tesseract." << std::endl;
            exit(1);
        }
        ocrNumbers->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
        ocrNumbers->SetVariable("tessedit_char_whitelist", "0123456789.%/()");
    }

    ~OCRReader() {
        if (ocrGeneral) {
            ocrGeneral->End();
            delete ocrGeneral;
        }
        if (ocrNumbers) {
            ocrNumbers->End();
            delete ocrNumbers;
        }
    }

    std::string performGeneralOCR(const cv::Mat& image) {
        ocrGeneral->SetImage(image.data, image.cols, image.rows, 1, image.cols);
        char* outText = ocrGeneral->GetUTF8Text();
        std::string result(outText);
        delete[] outText;
        return result;
    }
    
    std::string performNumbersOCR(const cv::Mat& image) {
        ocrNumbers->SetImage(image.data, image.cols, image.rows, 1, image.cols);
        char* outText = ocrNumbers->GetUTF8Text();
        std::string result(outText);
        delete[] outText;
        return result;
    }

    cv::Mat preprocessForNumbers(const cv::Mat& image) {
        cv::Mat gray, processed;

        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // For dark background with light text, we need to invert
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        
        // Try both normal and inverted thresholding
        cv::Mat thresh1, thresh2;
        cv::threshold(gray, thresh1, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        cv::threshold(inverted, thresh2, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        // Choose the one with more white pixels (likely better for OCR)
        int white1 = cv::countNonZero(thresh1);
        int white2 = cv::countNonZero(thresh2);
        processed = (white2 > white1) ? thresh2 : thresh1;
        
        // Clean up with morphology
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);
        
        // Scale up for better recognition
        cv::resize(processed, processed, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);

        return processed;
    }

    cv::Mat preprocessForText(const cv::Mat& image) {
        cv::Mat gray, processed;

        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // For dark background with light text, invert
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        
        // Use adaptive threshold for text
        cv::adaptiveThreshold(inverted, processed, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, 11, 2);
        
        // Scale up
        cv::resize(processed, processed, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);

        return processed;
    }

    cv::Mat preprocessImageForTeamNames(const cv::Mat& image) {
        cv::Mat gray, processed;

        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        int headerHeight = std::min(150, gray.rows / 3);
        cv::Rect headerRegion(0, 0, gray.cols, headerHeight);
        cv::Mat header = gray(headerRegion);
        
        cv::GaussianBlur(header, header, cv::Size(1, 1), 0);
        cv::adaptiveThreshold(header, header, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, 15, 10);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 1));
        cv::morphologyEx(header, header, cv::MORPH_CLOSE, kernel);
        cv::resize(header, header, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);
        
        return header;
    }

    std::vector<std::string> splitLines(const std::string& text) {
        std::vector<std::string> lines;
        std::istringstream stream(text);
        std::string line;

        while (std::getline(stream, line)) {
            line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
            if (!line.empty()) {
                lines.push_back(line);
            }
        }
        return lines;
    }

    std::pair<std::string, std::string> extractTeamNames(const cv::Mat& image) {
        // Try multiple approaches for team name extraction
        
        // Approach 1: Header region
        cv::Mat headerProcessed = preprocessImageForTeamNames(image);
        std::string headerOcrText = performGeneralOCR(headerProcessed);
        std::vector<std::string> headerLines = splitLines(headerOcrText);
        
        std::string homeTeam = "", awayTeam = "";
        
        // Look for team names in header
        for (const auto& line : headerLines) {
            std::string upperLine = line;
            std::transform(upperLine.begin(), upperLine.end(), upperLine.begin(), ::toupper);
            
            std::regex twoTeamsPattern("\\b([A-Z]{4,15})\\b.*\\b([A-Z]{4,15})\\b");
            std::smatch teamMatches;
            if (std::regex_search(upperLine, teamMatches, twoTeamsPattern)) {
                std::string team1 = teamMatches[1].str();
                std::string team2 = teamMatches[2].str();
                
                std::set<std::string> headerWords = {"MATCH", "STATS", "STATISTIC", "STATISTICS"};
                if (headerWords.find(team1) == headerWords.end() && 
                    headerWords.find(team2) == headerWords.end() &&
                    team1 != team2) {
                    homeTeam = team1;
                    awayTeam = team2;
                    break;
                }
            }
        }
        
        // Convert to proper case
        if (!homeTeam.empty()) {
            std::transform(homeTeam.begin(), homeTeam.end(), homeTeam.begin(), ::tolower);
            homeTeam[0] = std::toupper(homeTeam[0]);
        }
        
        if (!awayTeam.empty()) {
            std::transform(awayTeam.begin(), awayTeam.end(), awayTeam.begin(), ::tolower);
            awayTeam[0] = std::toupper(awayTeam[0]);
        }
        
        return {homeTeam, awayTeam};
    }

    std::string extractBestNumber(const std::string& text, bool expectPercentage = false) {
        std::vector<std::string> candidates;
        
        // Clean the text first
        std::string cleanText = text;
        cleanText = std::regex_replace(cleanText, std::regex("\\s+"), " ");
        
        if (expectPercentage) {
            std::regex percentPattern(R"(\b(\d+(?:\.\d+)?)%\b)");
            std::sregex_iterator iter(cleanText.begin(), cleanText.end(), percentPattern);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                candidates.push_back(iter->str(1) + "%");
            }
        }
        
        // Look for decimal numbers
        if (candidates.empty()) {
            std::regex decimalPattern(R"(\b(\d+\.\d+)\b)");
            std::sregex_iterator iter(cleanText.begin(), cleanText.end(), decimalPattern);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                std::string num = iter->str(1);
                if (num.length() <= 6) {
                    candidates.push_back(num);
                }
            }
        }
        
        // Look for integers
        if (candidates.empty()) {
            std::regex intPattern(R"(\b(\d+)\b)");
            std::sregex_iterator iter(cleanText.begin(), cleanText.end(), intPattern);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                std::string num = iter->str(1);
                if (num.length() <= 4) {
                    candidates.push_back(num);
                }
            }
        }
        
        // Look for fractions/ratios in parentheses
        if (candidates.empty() && text.find('(') != std::string::npos) {
            std::regex fractionPattern(R"(\((\d+/\d+)\))");
            std::sregex_iterator iter(cleanText.begin(), cleanText.end(), fractionPattern);
            std::sregex_iterator end;
            
            for (; iter != end; ++iter) {
                candidates.push_back(iter->str(1));
            }
        }
        
        return candidates.empty() ? "" : candidates[0];
    }

    int parseNumber(const std::string& value) {
        // Extract just the integer part of a stat value
        std::regex numPattern(R"((\d+))");
        std::smatch match;
        if (std::regex_search(value, match, numPattern)) {
            return std::stoi(match[1].str());
        }
        return 0;
    }

    std::vector<MatchStatistic> extractStatisticsThreeColumn(const cv::Mat& image) {
        std::vector<MatchStatistic> statistics;
        
        // Skip header - start from where stats begin
        int headerHeight = image.rows / 6;
        int statsHeight = image.rows - headerHeight;
        
        // EXPANDED column boundaries to capture full text and numbers
        int leftColumnWidth = image.cols * 0.20;
        int rightColumnWidth = image.cols * 0.20;
        int leftColumnStart = 0;
        int leftColumnEnd = leftColumnWidth;
        int rightColumnStart = image.cols - rightColumnWidth;
        int rightColumnEnd = image.cols;
        int middleColumnStart = image.cols * 0.25;
        int middleColumnEnd = image.cols * 0.75;
        
        // Set to 19 rows
        int estimatedRows = 19;
        int actualRowHeight = statsHeight / estimatedRows;
        
        // Add padding to row height to ensure we capture full text
        int rowPadding = actualRowHeight / 4; // Add 25% padding
        
        std::cerr << "Image size: " << image.cols << "x" << image.rows << std::endl;
        std::cerr << "Header height: " << headerHeight << ", Stats height: " << statsHeight << std::endl;
        std::cerr << "Processing " << estimatedRows << " rows with height " << actualRowHeight << " (padding: " << rowPadding << ")" << std::endl;
        std::cerr << "Column boundaries: " << leftColumnStart << "-" << leftColumnEnd 
                  << ", " << middleColumnStart << "-" << middleColumnEnd 
                  << ", " << rightColumnStart << "-" << rightColumnEnd << std::endl;
        
        // Save full stats debug image for verification
        cv::Rect fullStatsRect(0, headerHeight, image.cols, statsHeight);
        cv::Mat fullStatsImage = image(fullStatsRect);
        cv::Mat fullStatsProcessed = preprocessForText(fullStatsImage);
        cv::imwrite("debug_full_stats.png", fullStatsProcessed);
        
        // Process each row
        for (int row = 0; row < estimatedRows; ++row) {
            int rowY = headerHeight + (row * actualRowHeight);
            // Add padding to row height, but cap at image boundary
            int rowHeight = std::min(actualRowHeight + rowPadding, image.rows - rowY);
            
            if (rowHeight <= 0 || rowY >= image.rows) break;
            
            std::cerr << "\n--- Processing Row " << row << " (y=" << rowY << ", h=" << rowHeight << ") ---" << std::endl;
            
            // Extract full row for context and debugging first
            cv::Rect fullRowRect(0, rowY, image.cols, rowHeight);
            cv::Mat fullRowImage = image(fullRowRect);
            cv::Mat fullRowProcessed = preprocessForText(fullRowImage);
            cv::imwrite("debug_full_row_" + std::to_string(row) + ".png", fullRowProcessed);
            
            // Get full row text to check for all stats
            std::string fullRowText = performGeneralOCR(fullRowProcessed);
            std::string lowerFullRowText = fullRowText;
            std::transform(lowerFullRowText.begin(), lowerFullRowText.end(), lowerFullRowText.begin(), ::tolower);
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("^\\s+|\\s+$"), "");
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("\\s+"), " ");
            
            std::cerr << "Full row text: '" << fullRowText << "'" << std::endl;
            
            // Extract middle column to identify the stat
            cv::Rect middleRect(middleColumnStart, rowY, middleColumnEnd - middleColumnStart, rowHeight);
            if (middleRect.x + middleRect.width > image.cols || middleRect.y + middleRect.height > image.rows) {
                std::cerr << "Middle rect out of bounds, skipping row" << std::endl;
                continue;
            }
            
            cv::Mat middleImage = image(middleRect);
            cv::Mat middleProcessed = preprocessForText(middleImage);
            std::string middleText = performGeneralOCR(middleProcessed);
            std::string lowerMiddleText = middleText;
            std::transform(lowerMiddleText.begin(), lowerMiddleText.end(), lowerMiddleText.begin(), ::tolower);
            
            // Clean up the middle text
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("^\\s+|\\s+$"), "");
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("\\s+"), " ");
            
            std::cerr << "Middle text: '" << middleText << "'" << std::endl;
            std::cerr << "Lower/clean: '" << lowerMiddleText << "'" << std::endl;
            
            // PRIORITY: Check middle column first, then fall back to full row
            std::string foundStatName;
            int maxMatchLength = 0;
            
            // First pass: Check middle column only
            for (const auto& statPair : statMappings) {
                size_t posMiddle = lowerMiddleText.find(statPair.first);
                if (posMiddle != std::string::npos && statPair.first.length() > maxMatchLength) {
                    foundStatName = statPair.second;
                    maxMatchLength = statPair.first.length();
                }
            }
            
            // Second pass: If nothing found in middle, check full row
            if (foundStatName.empty()) {
                for (const auto& statPair : statMappings) {
                    size_t posFull = lowerFullRowText.find(statPair.first);
                    if (posFull != std::string::npos && statPair.first.length() > maxMatchLength) {
                        foundStatName = statPair.second;
                        maxMatchLength = statPair.first.length();
                    }
                }
            }
            
            if (foundStatName.empty()) {
                std::cerr << "No matching stat found for this row" << std::endl;
                continue;
            }
            
            std::cerr << "*** Found stat: " << foundStatName << " ***" << std::endl;
            
            // Extract left column (home team)
            cv::Rect leftRect(leftColumnStart, rowY, leftColumnEnd, rowHeight);
            if (leftRect.x + leftRect.width > image.cols || leftRect.y + leftRect.height > image.rows) {
                std::cerr << "Left rect out of bounds, skipping" << std::endl;
                continue;
            }
            
            // Extract right column (away team)
            cv::Rect rightRect(rightColumnStart, rowY, rightColumnEnd - rightColumnStart, rowHeight);
            if (rightRect.x + rightRect.width > image.cols || rightRect.y + rightRect.height > image.rows) {
                std::cerr << "Right rect out of bounds, skipping" << std::endl;
                continue;
            }
            
            cv::Mat leftImage = image(leftRect);
            cv::Mat rightImage = image(rightRect);
            
            cv::Mat leftProcessed = preprocessForNumbers(leftImage);
            cv::Mat rightProcessed = preprocessForNumbers(rightImage);
            
            // Try OCR on the ORIGINAL images without heavy preprocessing
            cv::Mat leftOriginalGray, rightOriginalGray;
            if (leftImage.channels() > 1) {
                cv::cvtColor(leftImage, leftOriginalGray, cv::COLOR_BGR2GRAY);
            } else {
                leftOriginalGray = leftImage.clone();
            }
            if (rightImage.channels() > 1) {
                cv::cvtColor(rightImage, rightOriginalGray, cv::COLOR_BGR2GRAY);
            } else {
                rightOriginalGray = rightImage.clone();
            }
            
            // Resize original images
            cv::resize(leftOriginalGray, leftOriginalGray, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);
            cv::resize(rightOriginalGray, rightOriginalGray, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);
            
            std::string leftText = performNumbersOCR(leftProcessed);
            std::string rightText = performNumbersOCR(rightProcessed);
            
            // Try with less aggressive preprocessing
            std::string leftTextOriginal = performNumbersOCR(leftOriginalGray);
            std::string rightTextOriginal = performNumbersOCR(rightOriginalGray);
            
            // Also try general OCR for comparison
            std::string leftTextGeneral = performGeneralOCR(leftProcessed);
            std::string rightTextGeneral = performGeneralOCR(rightProcessed);
            
            std::cerr << "Left numbers OCR (processed): '" << leftText << "'" << std::endl;
            std::cerr << "Left numbers OCR (original): '" << leftTextOriginal << "'" << std::endl;
            std::cerr << "Left general OCR: '" << leftTextGeneral << "'" << std::endl;
            std::cerr << "Right numbers OCR (processed): '" << rightText << "'" << std::endl;
            std::cerr << "Right numbers OCR (original): '" << rightTextOriginal << "'" << std::endl;
            std::cerr << "Right general OCR: '" << rightTextGeneral << "'" << std::endl;
            
            bool expectPercentage = (foundStatName == "possession" || 
                                   foundStatName == "passes completed" ||
                                   foundStatName == "crosses completed" ||
                                   foundStatName == "tackles won" ||
                                   foundStatName == "headers won");
            
            // Try all OCR results and pick best
            std::string homeValue = extractBestNumber(leftText, expectPercentage);
            if (homeValue.empty()) homeValue = extractBestNumber(leftTextOriginal, expectPercentage);
            if (homeValue.empty()) homeValue = extractBestNumber(leftTextGeneral, expectPercentage);
            
            std::string awayValue = extractBestNumber(rightText, expectPercentage);
            if (awayValue.empty()) awayValue = extractBestNumber(rightTextOriginal, expectPercentage);
            if (awayValue.empty()) awayValue = extractBestNumber(rightTextGeneral, expectPercentage);
            
            // AGGRESSIVE FALLBACK: Try full row OCR if columns completely fail
            if ((homeValue.empty() || awayValue.empty()) && !foundStatName.empty()) {
                std::cerr << "Column OCR failed, trying full row extraction..." << std::endl;
                
                // Extract all numbers from the full row - use the FIRST and LAST numbers
                std::vector<std::string> allNumbers;
                std::regex numberPattern(R"(\b(\d+(?:\.\d+)?%?)\b)");
                std::sregex_iterator iter(fullRowText.begin(), fullRowText.end(), numberPattern);
                std::sregex_iterator end;
                
                for (; iter != end; ++iter) {
                    allNumbers.push_back(iter->str(1));
                }
                
                std::cerr << "Found " << allNumbers.size() << " numbers in full row: ";
                for (const auto& n : allNumbers) std::cerr << n << " ";
                std::cerr << std::endl;
                
                // Use FIRST and LAST numbers (leftmost and rightmost on the screen)
                if (allNumbers.size() >= 2) {
                    if (homeValue.empty()) homeValue = allNumbers[0];
                    if (awayValue.empty()) awayValue = allNumbers[allNumbers.size() - 1];
                    std::cerr << "Extracted from full row: Home=" << homeValue << ", Away=" << awayValue << std::endl;
                } else if (allNumbers.size() == 1) {
                    // Only one number found - can't determine which side
                    std::cerr << "Only one number found, cannot extract both values" << std::endl;
                }
            }
            
            std::cerr << "Final values: Home='" << homeValue << "', Away='" << awayValue << "'" << std::endl;
            
            // Save debug images for this row
            std::string leftDebug = "debug_3col_row_" + std::to_string(row) + "_left.png";
            std::string middleDebug = "debug_3col_row_" + std::to_string(row) + "_middle.png";
            std::string rightDebug = "debug_3col_row_" + std::to_string(row) + "_right.png";
            cv::imwrite(leftDebug, leftProcessed);
            cv::imwrite(middleDebug, middleProcessed);
            cv::imwrite(rightDebug, rightProcessed);
            
            // Accept stat even if one value is missing, use "0" as default
            if (homeValue.empty()) homeValue = "0";
            if (awayValue.empty()) awayValue = "0";
            
            MatchStatistic stat;
            stat.statName = foundStatName;
            stat.homeValue = homeValue;
            stat.awayValue = awayValue;
            statistics.push_back(stat);
            
            std::cerr << "*** ADDED: " << foundStatName << " = " << homeValue << " vs " << awayValue << " ***" << std::endl;
        }
        
        return statistics;
    }
    
    // Alternative approach - detect rows dynamically rather than using fixed grid
    std::vector<MatchStatistic> extractStatisticsByDetection(const cv::Mat& image) {
        std::vector<MatchStatistic> statistics;
        
        // Skip header - start from where stats begin
        int headerHeight = image.rows / 6;
        int statsHeight = image.rows - headerHeight;
        
        cv::Rect statsRect(0, headerHeight, image.cols, statsHeight);
        cv::Mat statsImage = image(statsRect);
        
        // Preprocess for horizontal line detection
        cv::Mat gray, edges;
        if (statsImage.channels() > 1) {
            cv::cvtColor(statsImage, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = statsImage.clone();
        }
        
        // Find horizontal edges
        cv::Canny(gray, edges, 50, 150);
        cv::Mat lines = cv::Mat::zeros(edges.size(), CV_8UC1);
        
        // Find horizontal lines using morphology
        cv::Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(gray.cols/30, 1));
        cv::morphologyEx(edges, lines, cv::MORPH_OPEN, horizontalStructure);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(lines, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Sort contours by y-coordinate
        std::sort(contours.begin(), contours.end(), 
            [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                return cv::boundingRect(c1).y < cv::boundingRect(c2).y;
            });
        
        // Define rough column positions
        int leftEnd = image.cols * 0.20;
        int middleStart = image.cols * 0.25;
        int middleEnd = image.cols * 0.75;
        int rightStart = image.cols * 0.80;
        
        // Process each row defined by horizontal lines
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::Rect rowBox = cv::boundingRect(contours[i]);
            
            // Skip very small rows
            if (rowBox.height < 5) continue;
            
            // Look ahead to the next row (if exists)
            int nextRowY = (i < contours.size() - 1) ? 
                cv::boundingRect(contours[i+1]).y : statsHeight;
            int rowHeight = nextRowY - rowBox.y - 5;
            
            // Ensure reasonable row height with padding
            if (rowHeight < 10 || rowHeight > 100) rowHeight = 30;
            rowHeight += 10; // Add padding to capture full text
            
            cv::Rect fullRowRect(0, rowBox.y + headerHeight, statsImage.cols, std::min(rowHeight, image.rows - (rowBox.y + headerHeight)));
            cv::Mat fullRowImage = image(fullRowRect);
            
            // Get full row text to check for all stats
            cv::Mat fullRowProcessed = preprocessForText(fullRowImage);
            std::string fullRowText = performGeneralOCR(fullRowProcessed);
            std::string lowerFullRowText = fullRowText;
            std::transform(lowerFullRowText.begin(), lowerFullRowText.end(), lowerFullRowText.begin(), ::tolower);
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("^\\s+|\\s+$"), "");
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("\\s+"), " ");
            
            // Define column areas for this row
            cv::Rect leftRect(0, 0, leftEnd, fullRowImage.rows);
            cv::Rect middleRect(middleStart, 0, middleEnd - middleStart, fullRowImage.rows);
            cv::Rect rightRect(rightStart, 0, fullRowImage.cols - rightStart, fullRowImage.rows);
            
            // Extract and process middle region to identify stat name
            cv::Mat middleImage = fullRowImage(middleRect);
            cv::Mat middleProcessed = preprocessForText(middleImage);
            std::string middleText = performGeneralOCR(middleProcessed);
            std::string lowerMiddleText = middleText;
            std::transform(lowerMiddleText.begin(), lowerMiddleText.end(), lowerMiddleText.begin(), ::tolower);
            
            // Clean up text
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("^\\s+|\\s+$"), "");
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("\\s+"), " ");
            
            // PRIORITY: Check middle column first, then fall back to full row
            std::string foundStatName;
            int maxMatchLength = 0;
            
            // First pass: Check middle column only
            for (const auto& statPair : statMappings) {
                size_t posMiddle = lowerMiddleText.find(statPair.first);
                if (posMiddle != std::string::npos && statPair.first.length() > maxMatchLength) {
                    foundStatName = statPair.second;
                    maxMatchLength = statPair.first.length();
                }
            }
            
            // Second pass: If nothing found in middle, check full row
            if (foundStatName.empty()) {
                for (const auto& statPair : statMappings) {
                    size_t posFull = lowerFullRowText.find(statPair.first);
                    if (posFull != std::string::npos && statPair.first.length() > maxMatchLength) {
                        foundStatName = statPair.second;
                        maxMatchLength = statPair.first.length();
                    }
                }
            }
            
            // If found a valid stat, extract values
            if (!foundStatName.empty()) {
                // Extract and process left and right values
                cv::Mat leftImage = fullRowImage(leftRect);
                cv::Mat rightImage = fullRowImage(rightRect);
                
                cv::Mat leftProcessed = preprocessForNumbers(leftImage);
                cv::Mat rightProcessed = preprocessForNumbers(rightImage);
                
                // Try both specific and general OCR
                bool expectPercentage = (foundStatName == "possession" || 
                                       foundStatName == "passes completed" ||
                                       foundStatName == "crosses completed" ||
                                       foundStatName == "tackles won" ||
                                       foundStatName == "headers won");
                
                std::string leftText = performNumbersOCR(leftProcessed);
                std::string rightText = performNumbersOCR(rightProcessed);
                std::string leftTextGeneral = performGeneralOCR(leftProcessed);
                std::string rightTextGeneral = performGeneralOCR(rightProcessed);
                
                std::string homeValue = extractBestNumber(leftText, expectPercentage);
                if (homeValue.empty()) homeValue = extractBestNumber(leftTextGeneral, expectPercentage);
                
                std::string awayValue = extractBestNumber(rightText, expectPercentage);
                if (awayValue.empty()) awayValue = extractBestNumber(rightTextGeneral, expectPercentage);
                
                // AGGRESSIVE FALLBACK: Try full row if columns fail - use FIRST and LAST numbers
                if (homeValue.empty() || awayValue.empty()) {
                    std::vector<std::string> allNumbers;
                    std::regex numberPattern(R"(\b(\d+(?:\.\d+)?%?)\b)");
                    std::sregex_iterator iter(fullRowText.begin(), fullRowText.end(), numberPattern);
                    std::sregex_iterator end;
                    
                    for (; iter != end; ++iter) {
                        allNumbers.push_back(iter->str(1));
                    }
                    
                    if (allNumbers.size() >= 2) {
                        if (homeValue.empty()) homeValue = allNumbers[0];
                        if (awayValue.empty()) awayValue = allNumbers[allNumbers.size() - 1];
                    }
                }
                
                // Accept stat even if one value is missing
                if (homeValue.empty()) homeValue = "0";
                if (awayValue.empty()) awayValue = "0";
                
                MatchStatistic stat;
                stat.statName = foundStatName;
                stat.homeValue = homeValue;
                stat.awayValue = awayValue;
                statistics.push_back(stat);
                
                // Save debug images
                cv::imwrite("debug_row_" + std::to_string(i) + "_full.png", fullRowImage);
                cv::imwrite("debug_row_" + std::to_string(i) + "_left.png", leftProcessed);
                cv::imwrite("debug_row_" + std::to_string(i) + "_middle.png", middleProcessed);
                cv::imwrite("debug_row_" + std::to_string(i) + "_right.png", rightProcessed);
            }
        }
        
        return statistics;
    }

    void processScreenshot(const std::string& imagePath) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "ERROR: Could not load image: " << imagePath << std::endl;
            return;
        }

        std::cerr << "Image size: " << image.cols << "x" << image.rows << std::endl;

        // Extract team names
        auto teamNames = extractTeamNames(image);
        
        // Try both methods and choose the one with more results
        std::vector<MatchStatistic> stats1 = extractStatisticsThreeColumn(image);
        std::vector<MatchStatistic> stats2 = extractStatisticsByDetection(image);
        
        std::vector<MatchStatistic> stats = (stats1.size() >= stats2.size()) ? stats1 : stats2;
        
        // POST-PROCESSING: Calculate Total Shots from components
        std::string onTargetHome = "0", onTargetAway = "0";
        std::string offTargetHome = "0", offTargetAway = "0";
        
        for (const auto& stat : stats) {
            if (stat.statName == "on target") {
                onTargetHome = stat.homeValue;
                onTargetAway = stat.awayValue;
            } else if (stat.statName == "off target") {
                offTargetHome = stat.homeValue;
                offTargetAway = stat.awayValue;
            }
        }
        
        // Calculate total shots
        int totalShotsHome = parseNumber(onTargetHome) + parseNumber(offTargetHome);
        int totalShotsAway = parseNumber(onTargetAway) + parseNumber(offTargetAway);
        
        // Add Total Shots stat at the beginning
        MatchStatistic totalShotsStat;
        totalShotsStat.statName = "shots";
        totalShotsStat.homeValue = std::to_string(totalShotsHome);
        totalShotsStat.awayValue = std::to_string(totalShotsAway);
        totalShotsStat.homeTeam = teamNames.first.empty() ? "Home" : teamNames.first;
        totalShotsStat.awayTeam = teamNames.second.empty() ? "Away" : teamNames.second;
        
        // Insert at beginning
        stats.insert(stats.begin(), totalShotsStat);
        
        // Set team names for all other stats
        for (auto& stat : stats) {
            if (stat.homeTeam.empty()) {
                stat.homeTeam = teamNames.first.empty() ? "Home" : teamNames.first;
            }
            if (stat.awayTeam.empty()) {
                stat.awayTeam = teamNames.second.empty() ? "Away" : teamNames.second;
            }
        }

        // Output team names
        std::cout << "HOME_TEAM:" << (teamNames.first.empty() ? "UNKNOWN" : teamNames.first) << std::endl;
        std::cout << "AWAY_TEAM:" << (teamNames.second.empty() ? "UNKNOWN" : teamNames.second) << std::endl;

        // Output statistics
        for (const auto& stat : stats) {
            std::cout << "STAT:" << stat.statName << "|" << stat.homeValue << "|" << stat.awayValue << std::endl;
        }

        std::cerr << "Total stats extracted: " << stats.size() << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <screenshot_path>" << std::endl;
        return 1;
    }

    try {
        OCRReader reader;
        reader.processScreenshot(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}