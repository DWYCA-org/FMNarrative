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
#include <memory>
#include <stdexcept>

struct MatchStatistic {
    std::string statName;
    std::string homeValue;
    std::string awayValue;
    std::string homeTeam;
    std::string awayTeam;
};

// Main OCR engine that processes screenshots and extracts match statistics
class OCRReader {
private:
    std::unique_ptr<tesseract::TessBaseAPI> ocrGeneral; // For general text recognition
    std::unique_ptr<tesseract::TessBaseAPI> ocrNumbers; // For numbers 
    
    // Maps various stat name variations to standardized names
    std::map<std::string, std::string> statMappings = {
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
    // Initialize two Tesseract instances: one for general text, one for numbers
    OCRReader() {
        // Initialize general OCR
        ocrGeneral = std::make_unique<tesseract::TessBaseAPI>();
        if (ocrGeneral->Init(NULL, "eng")) {
            throw std::runtime_error("Could not initialize general tesseract.");
        }
        ocrGeneral->SetPageSegMode(tesseract::PSM_AUTO);
        
        // Initialize numbers-only OCR
        ocrNumbers = std::make_unique<tesseract::TessBaseAPI>();
        if (ocrNumbers->Init(NULL, "eng")) {
            // Clean up general OCR before throwing
            ocrGeneral->End();
            throw std::runtime_error("Could not initialize numbers tesseract.");
        }
        ocrNumbers->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
        ocrNumbers->SetVariable("tessedit_char_whitelist", "0123456789.%/()");
    }

    // Clean up Tesseract instances
    ~OCRReader() {
        if (ocrGeneral) {
            ocrGeneral->End();
        }
        if (ocrNumbers) {
            ocrNumbers->End();
        }
    }

    // Prevent copying to avoid double-delete issues
    OCRReader(const OCRReader&) = delete;
    OCRReader& operator=(const OCRReader&) = delete;

    // Allow moving
    OCRReader(OCRReader&&) noexcept = default;
    OCRReader& operator=(OCRReader&&) noexcept = default;

    // Perform OCR using general text recognition engine
    std::string performGeneralOCR(const cv::Mat& image) {
        ocrGeneral->SetImage(image.data, image.cols, image.rows, 1, image.cols);
        char* outText = ocrGeneral->GetUTF8Text();
        std::string result(outText);
        delete[] outText;
        return result;
    }
    
    // Perform OCR using numbers-only recognition engine
    std::string performNumbersOCR(const cv::Mat& image) {
        ocrNumbers->SetImage(image.data, image.cols, image.rows, 1, image.cols);
        char* outText = ocrNumbers->GetUTF8Text();
        std::string result(outText);
        delete[] outText;
        return result;
    }

    // Preprocess image regions containing numeric values using thresholding and scaling
    cv::Mat preprocessForNumbers(const cv::Mat& image) {
        cv::Mat gray, processed;

        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Try both normal and inverted thresholding
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        
        cv::Mat thresh1, thresh2;
        cv::threshold(gray, thresh1, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        cv::threshold(inverted, thresh2, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        // Choose the threshold with more white pixels (likely better for OCR)
        int white1 = cv::countNonZero(thresh1);
        int white2 = cv::countNonZero(thresh2);
        processed = (white2 > white1) ? thresh2 : thresh1;
        
        // Clean up noise with morphology
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);
        
        // Scale up for better recognition
        cv::resize(processed, processed, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);

        return processed;
    }

    // Preprocess regions containing text using adaptive thresholding
    cv::Mat preprocessForText(const cv::Mat& image) {
        cv::Mat gray, processed;

        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Invert for dark background with light text
        cv::Mat inverted;
        cv::bitwise_not(gray, inverted);
        
        // Use adaptive threshold for text
        cv::adaptiveThreshold(inverted, processed, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, 11, 2);
        
        // Scale up
        cv::resize(processed, processed, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);

        return processed;
    }

    // Preprocess the header region of the screenshot to extract team names
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

    // Split OCR text output into individual lines and trim whitespace
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

    // Extract home and away team names from the screenshot header
    std::pair<std::string, std::string> extractTeamNames(const cv::Mat& image) {
        cv::Mat headerProcessed = preprocessImageForTeamNames(image);
        std::string headerOcrText = performGeneralOCR(headerProcessed);
        std::vector<std::string> headerLines = splitLines(headerOcrText);
        
        std::string homeTeam = "", awayTeam = "";
        
        // Look for two capitalized words (team names) in the header
        for (const auto& line : headerLines) {
            std::string upperLine = line;
            std::transform(upperLine.begin(), upperLine.end(), upperLine.begin(), ::toupper);
            
            std::regex twoTeamsPattern("\\b([A-Z]{4,15})\\b.*\\b([A-Z]{4,15})\\b");
            std::smatch teamMatches;
            if (std::regex_search(upperLine, teamMatches, twoTeamsPattern)) {
                std::string team1 = teamMatches[1].str();
                std::string team2 = teamMatches[2].str();
                
                // Filter out common header words like "MATCH" or "STATS"
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
        
        // Convert from UPPERCASE to Proper Case
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

    // Extract the best numeric value from OCR text, handling percentages, decimals, and fractions
    std::string extractBestNumber(const std::string& text, bool expectPercentage = false) {
        std::vector<std::string> candidates;
        
        std::string cleanText = text;
        cleanText = std::regex_replace(cleanText, std::regex("\\s+"), " ");
        
        // Try to find percentage values first if expected
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
        
        // Look for fractions in parentheses (e.g., "12 (5/10)")
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

    // Parse integer from a stat value string
    int parseNumber(const std::string& value) {
        std::regex numPattern(R"((\d+))");
        std::smatch match;
        if (std::regex_search(value, match, numPattern)) {
            return std::stoi(match[1].str());
        }
        return 0;
    }

    // Extract statistics by dividing screenshot into 3 columns (home value | stat name | away value)
    std::vector<MatchStatistic> extractStatisticsThreeColumn(const cv::Mat& image) {
        std::vector<MatchStatistic> statistics;
        
        // Skip header, process only the stats area
        int headerHeight = image.rows / 6;
        int statsHeight = image.rows - headerHeight;
        
        // Define column boundaries for home stats, stat names, and away stats
        int leftColumnWidth = image.cols * 0.20;
        int rightColumnWidth = image.cols * 0.20;
        int leftColumnStart = 0;
        int leftColumnEnd = leftColumnWidth;
        int rightColumnStart = image.cols - rightColumnWidth;
        int rightColumnEnd = image.cols;
        int middleColumnStart = image.cols * 0.25;
        int middleColumnEnd = image.cols * 0.75;
        
        // Process exactly 19 rows (standard FM match stats screen)
        int estimatedRows = 19;
        int actualRowHeight = statsHeight / estimatedRows;
        int rowPadding = actualRowHeight / 4;
        
        std::cerr << "Image size: " << image.cols << "x" << image.rows << std::endl;
        std::cerr << "Header height: " << headerHeight << ", Stats height: " << statsHeight << std::endl;
        std::cerr << "Processing " << estimatedRows << " rows with height " << actualRowHeight << " (padding: " << rowPadding << ")" << std::endl;
        std::cerr << "Column boundaries: " << leftColumnStart << "-" << leftColumnEnd 
                  << ", " << middleColumnStart << "-" << middleColumnEnd 
                  << ", " << rightColumnStart << "-" << rightColumnEnd << std::endl;
        
        // Save debug image of entire stats area
        cv::Rect fullStatsRect(0, headerHeight, image.cols, statsHeight);
        cv::Mat fullStatsImage = image(fullStatsRect);
        cv::Mat fullStatsProcessed = preprocessForText(fullStatsImage);
        cv::imwrite("debug_full_stats.png", fullStatsProcessed);
        
        // Process each row to extract one statistic
        for (int row = 0; row < estimatedRows; ++row) {
            int rowY = headerHeight + (row * actualRowHeight);
            int rowHeight = std::min(actualRowHeight + rowPadding, image.rows - rowY);
            
            if (rowHeight <= 0 || rowY >= image.rows) break;
            
            std::cerr << "\n--- Processing Row " << row << " (y=" << rowY << ", h=" << rowHeight << ") ---" << std::endl;
            
            // Extract and OCR the full row for context
            cv::Rect fullRowRect(0, rowY, image.cols, rowHeight);
            cv::Mat fullRowImage = image(fullRowRect);
            cv::Mat fullRowProcessed = preprocessForText(fullRowImage);
            cv::imwrite("debug_full_row_" + std::to_string(row) + ".png", fullRowProcessed);
            
            std::string fullRowText = performGeneralOCR(fullRowProcessed);
            std::string lowerFullRowText = fullRowText;
            std::transform(lowerFullRowText.begin(), lowerFullRowText.end(), lowerFullRowText.begin(), ::tolower);
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("^\\s+|\\s+$"), "");
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("\\s+"), " ");
            
            std::cerr << "Full row text: '" << fullRowText << "'" << std::endl;
            
            // Extract middle column to identify the stat name
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
            
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("^\\s+|\\s+$"), "");
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("\\s+"), " ");
            
            std::cerr << "Middle text: '" << middleText << "'" << std::endl;
            std::cerr << "Lower/clean: '" << lowerMiddleText << "'" << std::endl;
            
            // Match OCR text to known stat names, prioritizing middle column
            std::string foundStatName;
            int maxMatchLength = 0;
            
            // Check middle column first
            for (const auto& statPair : statMappings) {
                size_t posMiddle = lowerMiddleText.find(statPair.first);
                if (posMiddle != std::string::npos && statPair.first.length() > maxMatchLength) {
                    foundStatName = statPair.second;
                    maxMatchLength = statPair.first.length();
                }
            }
            
            // Fallback to full row if middle column OCR failed
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
            
            // Extract left and right columns for home and away values
            cv::Rect leftRect(leftColumnStart, rowY, leftColumnEnd, rowHeight);
            if (leftRect.x + leftRect.width > image.cols || leftRect.y + leftRect.height > image.rows) {
                std::cerr << "Left rect out of bounds, skipping" << std::endl;
                continue;
            }
            
            cv::Rect rightRect(rightColumnStart, rowY, rightColumnEnd - rightColumnStart, rowHeight);
            if (rightRect.x + rightRect.width > image.cols || rightRect.y + rightRect.height > image.rows) {
                std::cerr << "Right rect out of bounds, skipping" << std::endl;
                continue;
            }
            
            cv::Mat leftImage = image(leftRect);
            cv::Mat rightImage = image(rightRect);
            
            cv::Mat leftProcessed = preprocessForNumbers(leftImage);
            cv::Mat rightProcessed = preprocessForNumbers(rightImage);
            
            // Try multiple preprocessing approaches for better OCR accuracy
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
            
            cv::resize(leftOriginalGray, leftOriginalGray, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);
            cv::resize(rightOriginalGray, rightOriginalGray, cv::Size(), 3.0, 3.0, cv::INTER_CUBIC);
            
            std::string leftText = performNumbersOCR(leftProcessed);
            std::string rightText = performNumbersOCR(rightProcessed);
            std::string leftTextOriginal = performNumbersOCR(leftOriginalGray);
            std::string rightTextOriginal = performNumbersOCR(rightOriginalGray);
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
            
            // Try all OCR results and pick the best number
            std::string homeValue = extractBestNumber(leftText, expectPercentage);
            if (homeValue.empty()) homeValue = extractBestNumber(leftTextOriginal, expectPercentage);
            if (homeValue.empty()) homeValue = extractBestNumber(leftTextGeneral, expectPercentage);
            
            std::string awayValue = extractBestNumber(rightText, expectPercentage);
            if (awayValue.empty()) awayValue = extractBestNumber(rightTextOriginal, expectPercentage);
            if (awayValue.empty()) awayValue = extractBestNumber(rightTextGeneral, expectPercentage);
            
            // Fallback: extract first and last numbers from full row text
            if ((homeValue.empty() || awayValue.empty()) && !foundStatName.empty()) {
                std::cerr << "Column OCR failed, trying full row extraction..." << std::endl;
                
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
                
                if (allNumbers.size() >= 2) {
                    if (homeValue.empty()) homeValue = allNumbers[0];
                    if (awayValue.empty()) awayValue = allNumbers[allNumbers.size() - 1];
                    std::cerr << "Extracted from full row: Home=" << homeValue << ", Away=" << awayValue << std::endl;
                } else if (allNumbers.size() == 1) {
                    std::cerr << "Only one number found, cannot extract both values" << std::endl;
                }
            }
            
            std::cerr << "Final values: Home='" << homeValue << "', Away='" << awayValue << "'" << std::endl;
            
            // Save debug images for manual verification
            std::string leftDebug = "debug_3col_row_" + std::to_string(row) + "_left.png";
            std::string middleDebug = "debug_3col_row_" + std::to_string(row) + "_middle.png";
            std::string rightDebug = "debug_3col_row_" + std::to_string(row) + "_right.png";
            cv::imwrite(leftDebug, leftProcessed);
            cv::imwrite(middleDebug, middleProcessed);
            cv::imwrite(rightDebug, rightProcessed);
            
            // Use "0" as default if value extraction completely failed
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
    
    // Alternative extraction method using edge detection to find row boundaries dynamically
    std::vector<MatchStatistic> extractStatisticsByDetection(const cv::Mat& image) {
        std::vector<MatchStatistic> statistics;
        
        int headerHeight = image.rows / 6;
        int statsHeight = image.rows - headerHeight;
        
        cv::Rect statsRect(0, headerHeight, image.cols, statsHeight);
        cv::Mat statsImage = image(statsRect);
        
        // Detect horizontal lines separating stat rows
        cv::Mat gray, edges;
        if (statsImage.channels() > 1) {
            cv::cvtColor(statsImage, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = statsImage.clone();
        }
        
        cv::Canny(gray, edges, 50, 150);
        cv::Mat lines = cv::Mat::zeros(edges.size(), CV_8UC1);
        
        cv::Mat horizontalStructure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(gray.cols/30, 1));
        cv::morphologyEx(edges, lines, cv::MORPH_OPEN, horizontalStructure);
        
        // Find and sort contours by vertical position
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(lines, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::sort(contours.begin(), contours.end(), 
            [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                return cv::boundingRect(c1).y < cv::boundingRect(c2).y;
            });
        
        int leftEnd = image.cols * 0.20;
        int middleStart = image.cols * 0.25;
        int middleEnd = image.cols * 0.75;
        int rightStart = image.cols * 0.80;
        
        // Process each detected row
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::Rect rowBox = cv::boundingRect(contours[i]);
            
            if (rowBox.height < 5) continue;
            
            // Calculate row height based on distance to next row
            int nextRowY = (i < contours.size() - 1) ? 
                cv::boundingRect(contours[i+1]).y : statsHeight;
            int rowHeight = nextRowY - rowBox.y - 5;
            
            if (rowHeight < 10 || rowHeight > 100) rowHeight = 30;
            rowHeight += 10;
            
            cv::Rect fullRowRect(0, rowBox.y + headerHeight, statsImage.cols, std::min(rowHeight, image.rows - (rowBox.y + headerHeight)));
            cv::Mat fullRowImage = image(fullRowRect);
            
            // OCR full row and extract stat name
            cv::Mat fullRowProcessed = preprocessForText(fullRowImage);
            std::string fullRowText = performGeneralOCR(fullRowProcessed);
            std::string lowerFullRowText = fullRowText;
            std::transform(lowerFullRowText.begin(), lowerFullRowText.end(), lowerFullRowText.begin(), ::tolower);
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("^\\s+|\\s+$"), "");
            lowerFullRowText = std::regex_replace(lowerFullRowText, std::regex("\\s+"), " ");
            
            // Define and extract column regions within this row
            cv::Rect leftRect(0, 0, leftEnd, fullRowImage.rows);
            cv::Rect middleRect(middleStart, 0, middleEnd - middleStart, fullRowImage.rows);
            cv::Rect rightRect(rightStart, 0, fullRowImage.cols - rightStart, fullRowImage.rows);
            
            cv::Mat middleImage = fullRowImage(middleRect);
            cv::Mat middleProcessed = preprocessForText(middleImage);
            std::string middleText = performGeneralOCR(middleProcessed);
            std::string lowerMiddleText = middleText;
            std::transform(lowerMiddleText.begin(), lowerMiddleText.end(), lowerMiddleText.begin(), ::tolower);
            
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("^\\s+|\\s+$"), "");
            lowerMiddleText = std::regex_replace(lowerMiddleText, std::regex("\\s+"), " ");
            
            // Match stat name from middle column or full row
            std::string foundStatName;
            int maxMatchLength = 0;
            
            for (const auto& statPair : statMappings) {
                size_t posMiddle = lowerMiddleText.find(statPair.first);
                if (posMiddle != std::string::npos && statPair.first.length() > maxMatchLength) {
                    foundStatName = statPair.second;
                    maxMatchLength = statPair.first.length();
                }
            }
            
            if (foundStatName.empty()) {
                for (const auto& statPair : statMappings) {
                    size_t posFull = lowerFullRowText.find(statPair.first);
                    if (posFull != std::string::npos && statPair.first.length() > maxMatchLength) {
                        foundStatName = statPair.second;
                        maxMatchLength = statPair.first.length();
                    }
                }
            }
            
            // Extract numeric values from left and right columns
            if (!foundStatName.empty()) {
                cv::Mat leftImage = fullRowImage(leftRect);
                cv::Mat rightImage = fullRowImage(rightRect);
                
                cv::Mat leftProcessed = preprocessForNumbers(leftImage);
                cv::Mat rightProcessed = preprocessForNumbers(rightImage);
                
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
                
                // Fallback: use first and last numbers from full row
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
                
                if (homeValue.empty()) homeValue = "0";
                if (awayValue.empty()) awayValue = "0";
                
                MatchStatistic stat;
                stat.statName = foundStatName;
                stat.homeValue = homeValue;
                stat.awayValue = awayValue;
                statistics.push_back(stat);
                
                // Save debug images for verification
                cv::imwrite("debug_row_" + std::to_string(i) + "_full.png", fullRowImage);
                cv::imwrite("debug_row_" + std::to_string(i) + "_left.png", leftProcessed);
                cv::imwrite("debug_row_" + std::to_string(i) + "_middle.png", middleProcessed);
                cv::imwrite("debug_row_" + std::to_string(i) + "_right.png", rightProcessed);
            }
        }
        
        return statistics;
    }

    // Main processing function: extracts team names and all statistics from screenshot
    void processScreenshot(const std::string& imagePath) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            throw std::runtime_error("Could not load image: " + imagePath);
        }

        std::cerr << "Image size: " << image.cols << "x" << image.rows << std::endl;

        auto teamNames = extractTeamNames(image);
        
        // Try both extraction methods and use the one with more results
        std::vector<MatchStatistic> stats1 = extractStatisticsThreeColumn(image);
        std::vector<MatchStatistic> stats2 = extractStatisticsByDetection(image);
        
        std::vector<MatchStatistic> stats = (stats1.size() >= stats2.size()) ? stats1 : stats2;
        
        // Calculate total shots by adding on-target and off-target shots
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
        
        int totalShotsHome = parseNumber(onTargetHome) + parseNumber(offTargetHome);
        int totalShotsAway = parseNumber(onTargetAway) + parseNumber(offTargetAway);
        
        // Add total shots as first statistic
        MatchStatistic totalShotsStat;
        totalShotsStat.statName = "shots";
        totalShotsStat.homeValue = std::to_string(totalShotsHome);
        totalShotsStat.awayValue = std::to_string(totalShotsAway);
        totalShotsStat.homeTeam = teamNames.first.empty() ? "Home" : teamNames.first;
        totalShotsStat.awayTeam = teamNames.second.empty() ? "Away" : teamNames.second;
        
        stats.insert(stats.begin(), totalShotsStat);
        
        // Set team names for all statistics
        for (auto& stat : stats) {
            if (stat.homeTeam.empty()) {
                stat.homeTeam = teamNames.first.empty() ? "Home" : teamNames.first;
            }
            if (stat.awayTeam.empty()) {
                stat.awayTeam = teamNames.second.empty() ? "Away" : teamNames.second;
            }
        }

        // Output results in parseable format
        std::cout << "HOME_TEAM:" << (teamNames.first.empty() ? "UNKNOWN" : teamNames.first) << std::endl;
        std::cout << "AWAY_TEAM:" << (teamNames.second.empty() ? "UNKNOWN" : teamNames.second) << std::endl;

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