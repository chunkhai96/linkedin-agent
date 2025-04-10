from typing import Optional
from linkedin_api.clients.restli.client import RestliClient

class LinkedInClient:
    def __init__(self, access_token: str):
        """Initialize LinkedIn client with access token."""
        self.access_token = access_token
        self.client = RestliClient()
        self.client.session.hooks["response"].append(lambda r: r.raise_for_status())
        
        # Fetch user profile to get person URN
        me_response = self.client.get(
            resource_path="/userinfo",
            access_token=self.access_token
        )
        # print(me_response.entity)
        self.person_urn = f"urn:li:person:{me_response.entity['sub']}"
    
    def post_content(self, text: str, use_legacy_api: bool = False) -> dict:
        """Post content to LinkedIn using either the new /posts or legacy /ugcPosts API.
        
        Args:
            text: The text content to post
            use_legacy_api: Whether to use the legacy /ugcPosts API instead of /posts
            
        Returns:
            Response from LinkedIn API containing the post ID
            
        Raises:
            Exception: If posting fails
        """
        try:
            if use_legacy_api:
                # Use legacy /ugcPosts endpoint
                response = self.client.create(
                    resource_path="/ugcPosts",
                    entity={
                        "author": self.person_urn,
                        "lifecycleState": "PUBLISHED",
                        "specificContent": {
                            "com.linkedin.ugc.ShareContent": {
                                "shareCommentary": {"text": text},
                                "shareMediaCategory": "NONE"
                            }
                        },
                        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
                    },
                    access_token=self.access_token
                )
            else:
                # Use new /posts endpoint
                response = self.client.create(
                    resource_path="/posts",
                    entity={
                        "author": self.person_urn,
                        "lifecycleState": "PUBLISHED",
                        "visibility": "PUBLIC",
                        "commentary": text,
                        "distribution": {
                            "feedDistribution": "MAIN_FEED",
                            "targetEntities": [],
                            "thirdPartyDistributionChannels": []
                        }
                    },
                    access_token=self.access_token
                )
            return f'https://www.linkedin.com/feed/update/{response.decoded_entity_id}'
        except Exception as e:
            raise Exception(f"Failed to post to LinkedIn: {str(e)}")